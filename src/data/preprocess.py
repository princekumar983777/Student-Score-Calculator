from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a ColumnTransformer that imputes, one-hot encodes categoricals and scales numerics."""
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_cols),
            ("numerical", numeric_transformer, numerical_cols),
        ],
        remainder="drop",
    )


def save_preprocessor(preprocessor: ColumnTransformer, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, p)


def load_preprocessor(path: str | Path) -> ColumnTransformer:
    return joblib.load(path)

