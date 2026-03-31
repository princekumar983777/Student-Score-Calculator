from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from src.data.preprocess import load_preprocessor
from src.models.ann_model import FullyConnectedRegressor, predict_ann


@dataclass(frozen=True)
class LoadedModel:
    model_type: str  # "sklearn" | "ann"
    model: Any


def load_best_model(models_dir: str | Path) -> LoadedModel:
    models_dir = Path(models_dir)
    sklearn_path = models_dir / "best_model.pkl"
    ann_path = models_dir / "best_model.pt"

    if sklearn_path.exists():
        return LoadedModel(model_type="sklearn", model=joblib.load(sklearn_path))
    if ann_path.exists():
        payload = torch.load(ann_path, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload and "input_dim" in payload:
            model = FullyConnectedRegressor(
                input_dim=int(payload["input_dim"]),
                hidden_layers=list(payload["hidden_layers"]),
                dropout=float(payload.get("dropout", 0.0)),
            )
            model.load_state_dict(payload["state_dict"])
            return LoadedModel(model_type="ann", model=model)
        # fallback: entire model
        return LoadedModel(model_type="ann", model=payload)
    raise FileNotFoundError(f"No best model found in {models_dir}. Expected best_model.pkl or best_model.pt")


def predict_from_dataframe(
    df: pd.DataFrame,
    *,
    preprocessor_path: str | Path,
    models_dir: str | Path,
    drop_cols: Optional[list[str]] = None,
    device: str = "auto",
) -> np.ndarray:
    drop_cols = drop_cols or []
    X = df.drop(columns=drop_cols, errors="ignore")
    preprocessor = load_preprocessor(preprocessor_path)
    Xp = preprocessor.transform(X)
    loaded = load_best_model(models_dir)

    if loaded.model_type == "sklearn":
        return loaded.model.predict(X)

    return predict_ann(loaded.model, Xp, device=device)

