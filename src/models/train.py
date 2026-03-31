from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.models.ann_model import AnnTrainResult, predict_ann, train_ann
from src.models.sklearn_models import create_sklearn_model
from src.utils.metrics import compute_regression_metrics


@dataclass(frozen=True)
class ModelRun:
    model_name: str
    model_type: str  # "sklearn" | "ann"
    model: Any
    val_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]] = None


def train_sklearn_model(
    model_name: str,
    model_params: Dict[str, Any],
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> ModelRun:
    reg = create_sklearn_model(model_name, model_params)
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", reg)])
    pipe.fit(X_train, y_train)
    val_pred = pipe.predict(X_val)
    val_metrics = compute_regression_metrics(y_val, val_pred).to_dict()
    return ModelRun(model_name=model_name, model_type="sklearn", model=pipe, val_metrics=val_metrics)


def train_pytorch_ann(
    model_params: Dict[str, Any],
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> tuple[ModelRun, Dict[str, Any]]:
    preprocessor.fit(X_train)
    Xtr = preprocessor.transform(X_train)
    Xva = preprocessor.transform(X_val)

    result: AnnTrainResult = train_ann(
        X_train=Xtr,
        y_train=y_train,
        X_val=Xva,
        y_val=y_val,
        hidden_layers=list(model_params["hidden_layers"]),
        dropout=float(model_params.get("dropout", 0.0)),
        batch_size=int(model_params["batch_size"]),
        epochs=int(model_params["epochs"]),
        learning_rate=float(model_params["learning_rate"]),
        weight_decay=float(model_params.get("weight_decay", 0.0)),
        patience=int(model_params.get("patience", 0)),
        device=str(model_params.get("device", "auto")),
    )

    val_pred = predict_ann(result.model, Xva, device=str(model_params.get("device", "auto")))
    val_metrics = compute_regression_metrics(y_val, val_pred).to_dict()

    meta = {
        "input_dim": int(Xtr.shape[1]),
        "hidden_layers": list(model_params["hidden_layers"]),
        "dropout": float(model_params.get("dropout", 0.0)),
        "device": str(model_params.get("device", "auto")),
        "epochs_trained": int(result.epochs_trained),
        "best_val_loss": float(result.best_val_loss),
    }

    return (
        ModelRun(model_name="ann", model_type="ann", model=result.model, val_metrics=val_metrics),
        meta,
    )

