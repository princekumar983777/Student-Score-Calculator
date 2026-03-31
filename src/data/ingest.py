from __future__ import annotations

from pathlib import Path

from src.utils.common import ensure_dir, read_yaml
from src.utils.logger import setup_logger


def run_ingestion(config_path: str = "configs/config.yaml") -> str:
    """Ingest raw data into `data/raw/`.

    For this repo, ingestion is a simple file copy from `research/` if present.
    """
    cfg = read_yaml(config_path)
    logger = setup_logger()
    raw_path = Path(cfg["paths"]["raw_data"])
    ensure_dir(raw_path.parent)

    if raw_path.exists():
        logger.info(f"Raw data already exists at {raw_path}")
        return str(raw_path)

    candidate = Path("research/student_habits_performance.csv")
    if not candidate.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path} and no fallback at {candidate}. "
            "Place your dataset at data/raw/students.csv."
        )

    raw_path.write_bytes(candidate.read_bytes())
    logger.info(f"Copied raw dataset from {candidate} to {raw_path}")
    return str(raw_path)

