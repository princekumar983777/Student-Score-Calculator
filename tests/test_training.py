from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocess import build_preprocessor
from src.models.train import train_sklearn_model


def test_train_sklearn_model_runs() -> None:
    X_train = pd.DataFrame(
        {
            "age": [17, 18, 19, 18, 17, 16],
            "gender": ["M", "F", "M", "F", "F", "M"],
            "study_hours_per_day": [1.0, 2.0, 3.0, 2.5, 1.2, 0.8],
        }
    )
    y_train = np.array([55, 60, 70, 66, 57, 50], dtype=float)
    X_val = pd.DataFrame(
        {
            "age": [18, 17],
            "gender": ["F", "M"],
            "study_hours_per_day": [2.1, 1.1],
        }
    )
    y_val = np.array([61, 54], dtype=float)

    run = train_sklearn_model(
        model_name="linear_regression",
        model_params={},
        preprocessor=build_preprocessor(X_train),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )
    assert "r2" in run.val_metrics

