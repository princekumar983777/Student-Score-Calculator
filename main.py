from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.pipelines.preprocessing_pipeline import run_preprocessing_pipeline
from src.pipelines.training_pipeline import run_training_pipeline
from src.utils.common import read_yaml
from src.utils.logger import setup_logger
from src.models.predict import predict_from_dataframe
from src.data.ingest import run_ingestion


def _cmd_preprocess(args: argparse.Namespace) -> None:
    run_preprocessing_pipeline(config_path=args.config)

def _cmd_ingest(args: argparse.Namespace) -> None:
    run_ingestion(config_path=args.config)


def _cmd_train(args: argparse.Namespace) -> None:
    # Ensure preprocessing outputs exist; if not, run preprocess first.
    cfg = read_yaml(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    needed = [processed_dir / "train.csv", processed_dir / "val.csv", processed_dir / "test.csv", Path(cfg["paths"]["preprocessor_path"])]
    if not all(p.exists() for p in needed):
        run_preprocessing_pipeline(config_path=args.config)
    run_training_pipeline(config_path=args.config)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    # Training pipeline writes `artifacts/metrics.json`. For now, evaluation just prints it.
    cfg = read_yaml(args.config)
    metrics_path = Path(cfg["paths"]["metrics_path"])
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics report not found at {metrics_path}. Run: python main.py train")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2))


def _cmd_predict(args: argparse.Namespace) -> None:
    cfg = read_yaml(args.config)
    df = pd.read_csv(args.input)
    drop_cols = list(cfg["data"].get("drop_cols", []))
    # if target exists in input, drop it
    if cfg["data"]["target_col"] in df.columns:
        df = df.drop(columns=[cfg["data"]["target_col"]])

    pred = predict_from_dataframe(
        df=df,
        preprocessor_path=cfg["paths"]["preprocessor_path"],
        models_dir=cfg["paths"]["models_dir"],
        drop_cols=drop_cols,
        device=str(cfg["models"]["ann"].get("device", "auto")),
    )
    out = pd.DataFrame({"prediction": pred})
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
    else:
        print(out.to_csv(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Student exam score prediction ML system")
    p.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    sub_ing = sub.add_parser("ingest", help="Ingest raw data into data/raw/")
    sub_ing.set_defaults(func=_cmd_ingest)

    sub_pre = sub.add_parser("preprocess", help="Run preprocessing (splits + preprocessor)")
    sub_pre.set_defaults(func=_cmd_preprocess)

    sub_train = sub.add_parser("train", help="Train and select best model")
    sub_train.set_defaults(func=_cmd_train)

    sub_eval = sub.add_parser("evaluate", help="Print metrics report JSON")
    sub_eval.set_defaults(func=_cmd_evaluate)

    sub_pred = sub.add_parser("predict", help="Run prediction on a CSV file")
    sub_pred.add_argument("--input", required=True, help="Input CSV file path")
    sub_pred.add_argument("--output", default=None, help="Optional output CSV path")
    sub_pred.set_defaults(func=_cmd_predict)

    return p


def main() -> None:
    setup_logger()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

