from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

from src.data.load_data import load_csv
from src.data.preprocess import build_preprocessor, load_preprocessor
from src.models.ann_model import FullyConnectedRegressor, predict_ann
from src.models.train import ModelRun, train_pytorch_ann, train_sklearn_model
from src.utils.common import ensure_dir, read_yaml, set_seed, write_json
from src.utils.logger import setup_logger


@dataclass(frozen=True)
class TrainingOutputs:
    best_model_name: str
    best_model_type: str
    best_model_path: str
    metrics: Dict[str, Any]


def _configure_mlflow(cfg: Dict[str, Any]) -> None:
    tracking_uri = cfg.get("mlflow", {}).get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.get("mlflow", {}).get("experiment_name", "default"))


def _register_and_stage_best(
    *,
    cfg: Dict[str, Any],
    best_run_id: str,
    artifact_subpath: str = "model",
) -> Optional[Dict[str, Any]]:
    mlflow_cfg = cfg.get("mlflow", {})
    if not mlflow_cfg.get("register_best_model", False):
        return None

    model_name = mlflow_cfg.get("registered_model_name", "best_model")
    model_uri = f"runs:/{best_run_id}/{artifact_subpath}"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    stage = mlflow_cfg.get("stage_after_register")
    if stage:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage=stage,
            archive_existing_versions=False,
        )
    return {"registered_model_name": model_name, "version": result.version, "stage": stage}


def run_training_pipeline(config_path: str = "configs/config.yaml") -> TrainingOutputs:
    cfg = read_yaml(config_path)
    logger = setup_logger()
    project_seed = int(cfg["project"]["random_seed"])
    set_seed(project_seed)

    paths = cfg["paths"]
    models_dir = ensure_dir(paths["models_dir"])

    processed_dir = Path(paths["processed_dir"])
    train_df = load_csv(processed_dir / "train.csv")
    val_df = load_csv(processed_dir / "val.csv")
    test_df = load_csv(processed_dir / "test.csv")
    target = cfg["data"]["target_col"]

    drop_cols = list(cfg["data"].get("drop_cols", []))
    X_train = train_df.drop(columns=[target, *drop_cols], errors="ignore")
    y_train = train_df[target].to_numpy()
    X_val = val_df.drop(columns=[target, *drop_cols], errors="ignore")
    y_val = val_df[target].to_numpy()
    X_test = test_df.drop(columns=[target, *drop_cols], errors="ignore")
    y_test = test_df[target].to_numpy()

    preprocessor = load_preprocessor(paths["preprocessor_path"])

    _configure_mlflow(cfg)

    enabled: List[str] = list(cfg["models"]["enabled"])
    compare_metric: str = str(cfg["training"]["compare_metric"])

    all_results: Dict[str, Any] = {}
    best = None
    best_run_id = None
    best_score = -float("inf")

    for model_name in enabled:
        logger.info(f"Training model={model_name}")
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.set_tag("model_name", model_name)
            mlflow.log_params({"random_seed": project_seed})

            if model_name == "ann":
                ann_run, ann_meta = train_pytorch_ann(
                    model_params=dict(cfg["models"]["ann"]),
                    preprocessor=build_preprocessor(X_train),
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                )
                score = float(ann_run.val_metrics.get(compare_metric, -float("inf")))
                mlflow.log_params({f"ann_{k}": v for k, v in ann_meta.items()})
                mlflow.log_metrics({f"val_{k}": float(v) for k, v in ann_run.val_metrics.items()})

                # log model
                mlflow.pytorch.log_model(ann_run.model, artifact_path="model")
                mlflow.log_artifact(paths["preprocessor_path"], artifact_path="artifacts")

                all_results[model_name] = {"val": ann_run.val_metrics, "meta": ann_meta, "run_id": run.info.run_id}
            else:
                params = dict(cfg["models"].get(model_name, {}))
                mlflow.log_params({f"{model_name}_{k}": v for k, v in params.items()})
                sk_run: ModelRun = train_sklearn_model(
                    model_name=model_name,
                    model_params=params,
                    preprocessor=build_preprocessor(X_train),
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                )
                score = float(sk_run.val_metrics.get(compare_metric, -float("inf")))
                mlflow.log_metrics({f"val_{k}": float(v) for k, v in sk_run.val_metrics.items()})
                mlflow.sklearn.log_model(sk_run.model, artifact_path="model")
                mlflow.log_artifact(paths["preprocessor_path"], artifact_path="artifacts")

                all_results[model_name] = {"val": sk_run.val_metrics, "run_id": run.info.run_id}

            if score > best_score:
                best_score = score
                best = model_name
                best_run_id = run.info.run_id

    if best is None or best_run_id is None:
        raise RuntimeError("No model was trained; check config.models.enabled")

    logger.info(f"Best model by val {compare_metric}: {best} ({best_score:.4f})")

    # Evaluate best on test and persist best artifact locally.
    if best == "ann":
        Xte = preprocessor.transform(X_test)
        ann_run, ann_meta = train_pytorch_ann(
            model_params=dict(cfg["models"]["ann"]),
            preprocessor=build_preprocessor(X_train),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )
        test_pred = predict_ann(ann_run.model, Xte, device=str(cfg["models"]["ann"].get("device", "auto")))
        from src.utils.metrics import compute_regression_metrics

        test_metrics = {k: float(v) for k, v in compute_regression_metrics(y_test, test_pred).to_dict().items()}

        pt_path = models_dir / "best_model.pt"
        torch.save(
            {
                "state_dict": ann_run.model.state_dict(),
                "input_dim": int(Xte.shape[1]),
                "hidden_layers": list(cfg["models"]["ann"]["hidden_layers"]),
                "dropout": float(cfg["models"]["ann"].get("dropout", 0.0)),
            },
            pt_path,
        )
        best_model_path = str(pt_path)
        all_results[best]["test"] = test_metrics
    else:
        # Refit best sklearn pipeline on train+val then test.
        preprocessor = build_preprocessor(pd.concat([X_train, X_val], axis=0))
        params = dict(cfg["models"].get(best, {}))
        from src.models.sklearn_models import create_sklearn_model
        from sklearn.pipeline import Pipeline

        reg = create_sklearn_model(best, params)
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", reg)])
        Xtrv = pd.concat([X_train, X_val], axis=0)
        ytrv = np.concatenate([y_train, y_val], axis=0)
        pipe.fit(Xtrv, ytrv)
        test_pred = pipe.predict(X_test)
        from src.utils.metrics import compute_regression_metrics

        test_metrics = compute_regression_metrics(y_test, test_pred).to_dict()
        pkl_path = models_dir / "best_model.pkl"
        joblib.dump(pipe, pkl_path)
        best_model_path = str(pkl_path)
        all_results[best]["test"] = test_metrics

    registry_info = _register_and_stage_best(cfg=cfg, best_run_id=best_run_id)
    report = {
        "best_model": {"name": best, "compare_metric": compare_metric, "val_score": best_score, "path": best_model_path},
        "models": all_results,
        "mlflow": {"best_run_id": best_run_id, "registry": registry_info},
    }
    write_json(paths["metrics_path"], report)
    logger.info(f"Saved metrics report to {paths['metrics_path']}")

    return TrainingOutputs(
        best_model_name=best,
        best_model_type="ann" if best == "ann" else "sklearn",
        best_model_path=best_model_path,
        metrics=report,
    )

