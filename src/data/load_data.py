from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load raw CSV into a pandas DataFrame."""
    return pd.read_csv(path)


def save_dataframe_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save dataframe as CSV (creates parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

