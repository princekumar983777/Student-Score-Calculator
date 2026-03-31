from __future__ import annotations

import pandas as pd

from src.data.preprocess import build_preprocessor
from src.features.build_features import split_features_target


def test_split_features_target_shapes() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            "age": [17, 18, 17, 19, 18, 17, 16, 18, 19, 17],
            "gender": ["M", "F", "M", "F", "F", "M", "M", "F", "M", "F"],
            "study_hours_per_day": [1.0, 2.0, 3.0, 1.5, 2.2, 3.1, 0.5, 2.9, 1.1, 4.0],
            "exam_score": [55, 60, 70, 58, 66, 75, 50, 72, 57, 80],
        }
    )
    split = split_features_target(
        df=df,
        target_col="exam_score",
        drop_cols=["student_id"],
        test_size=0.2,
        val_size=0.25,
        random_seed=42,
    )
    assert split.X_train.shape[0] > 0
    assert split.X_val.shape[0] > 0
    assert split.X_test.shape[0] > 0
    assert len(split.y_train) == split.X_train.shape[0]


def test_preprocessor_fit_transform() -> None:
    X = pd.DataFrame(
        {
            "age": [17, 18, 19],
            "gender": ["M", "F", "M"],
            "study_hours_per_day": [1.0, 2.0, 3.0],
        }
    )
    pre = build_preprocessor(X)
    pre.fit(X)
    Xt = pre.transform(X)
    assert Xt.shape[0] == 3

