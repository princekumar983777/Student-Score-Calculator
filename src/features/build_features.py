from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class FeatureSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def split_features_target(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: List[str],
    test_size: float,
    val_size: float,
    random_seed: int,
) -> FeatureSplit:
    """Split dataframe into train/val/test with reproducibility."""
    missing = [c for c in [target_col, *drop_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Target must be present; drop rows with missing target.
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    X = df.drop(columns=[target_col, *drop_cols], errors="ignore")
    y = df[target_col].to_numpy()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    # val_size is relative to trainval size
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=random_seed
    )
    return FeatureSplit(X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)

