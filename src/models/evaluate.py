from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.utils.metrics import RegressionMetrics, compute_regression_metrics


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    m = compute_regression_metrics(y_true=y_true, y_pred=y_pred)
    return m.to_dict()


def maybe_save_predictions(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Optional[str],
) -> None:
    if not path:
        return
    out = X.copy()
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out.to_csv(path, index=False)

