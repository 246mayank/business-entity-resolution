"""Paths and constants shared by the whole pipeline.

All paths are relative to the project root (the `student_resource/` folder that
holds `dataset/`). Override with the BER_ROOT environment variable when the code
runs from somewhere else, e.g. an AWS instance with the data on another disk.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parents[3]))

DATASET_DIR = ROOT / "dataset"
CACHE_DIR = ROOT / "cache"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)
SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")

SEED = 42

# Records are bucketed by their numeric id (S1-925783039 -> 925783039), which is
# deterministic across machines and library versions.
# Dev sample: records whose numeric id % 100 < SAMPLE_PCT.
SAMPLE_PCT = 5
# Validation fold: S1 entities whose (numeric id // 100) % VALID_MOD == 0 (20%).
# Uses different digits than the sample so the sample contains both folds.
VALID_MOD = 5


def raw_path(split: str, source: int) -> Path:
    return DATASET_DIR / split / f"{split}_source{source}.tsv"


def raw_gt_path() -> Path:
    return DATASET_DIR / "train" / "train_ground_truth.tsv"


def parquet_path(split: str, source: int) -> Path:
    return CACHE_DIR / f"{split}_source{source}.parquet"


def gt_pairs_path() -> Path:
    return CACHE_DIR / "train_gt_pairs.parquet"


def gt_s1_path() -> Path:
    return CACHE_DIR / "train_gt_s1.parquet"


def sample_dir(pct: int = SAMPLE_PCT) -> Path:
    return CACHE_DIR / f"sample_{pct}pct"
