from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.data.load_data import load_csv, save_dataframe_csv
from src.data.preprocess import build_preprocessor, save_preprocessor
from src.features.build_features import split_features_target
from src.utils.common import ensure_dir, read_yaml, set_seed
from src.utils.logger import setup_logger


def run_preprocessing_pipeline(config_path: str = "configs/config.yaml") -> Dict[str, str]:
    cfg = read_yaml(config_path)
    logger = setup_logger()

    seed = int(cfg["project"]["random_seed"])
    set_seed(seed)

    paths = cfg["paths"]
    raw_path = paths["raw_data"]
    processed_dir = ensure_dir(paths["processed_dir"])

    df = load_csv(raw_path)
    logger.info(f"Loaded raw dataset with shape={df.shape} from {raw_path}")

    split = split_features_target(
        df=df,
        target_col=cfg["data"]["target_col"],
        drop_cols=list(cfg["data"].get("drop_cols", [])),
        test_size=float(cfg["data"]["test_size"]),
        val_size=float(cfg["data"]["val_size"]),
        random_seed=seed,
    )

    target = cfg["data"]["target_col"]
    train_out = split.X_train.copy()
    train_out[target] = pd.Series(split.y_train, index=split.X_train.index)
    val_out = split.X_val.copy()
    val_out[target] = pd.Series(split.y_val, index=split.X_val.index)
    test_out = split.X_test.copy()
    test_out[target] = pd.Series(split.y_test, index=split.X_test.index)

    save_dataframe_csv(train_out.reset_index(drop=True), processed_dir / "train.csv")
    save_dataframe_csv(val_out.reset_index(drop=True), processed_dir / "val.csv")
    save_dataframe_csv(test_out.reset_index(drop=True), processed_dir / "test.csv")

    preprocessor = build_preprocessor(split.X_train)
    preprocessor.fit(split.X_train)
    save_preprocessor(preprocessor, paths["preprocessor_path"])
    logger.info(f"Saved preprocessor to {paths['preprocessor_path']}")

    return {
        "train_path": str(Path(processed_dir) / "train.csv"),
        "val_path": str(Path(processed_dir) / "val.csv"),
        "test_path": str(Path(processed_dir) / "test.csv"),
        "preprocessor_path": str(paths["preprocessor_path"]),
    }

