"""Interface contracts between pipeline stages (see docs/plan/00_combined_plan.md §3).

Stages exchange Parquet files in cache/. Producers write through the `write_*`
helpers (which validate first); consumers read through the `load_*` helpers and
may assume the contract holds. Extra columns are allowed; contract columns may
never be removed or renamed without a contract PR approved by both owners.

Datasets: "sample_5pct" (dev sample of train), "train", "test".
"""
import json
import time
from pathlib import Path

import polars as pl

from . import config
from . import io_utils as io
from . import sample

DATASETS = ("sample_5pct", "train", "test")

# C1 — normalised records, one file per (dataset, source).
C1_COLUMNS = {
    "entity_id": pl.String,
    "country": pl.String,
    "name_clean": pl.String,
    "name_core": pl.String,
    "legal_form": pl.String,
    "addr_clean": pl.String,
    "postcode": pl.String,
    "house_no": pl.String,
    "is_native_script": pl.Boolean,
}

# C2 — candidate pairs, one file per dataset.
C2_COLUMNS = {
    "s1_id": pl.String,
    "id": pl.String,
    "blk_name_sim": pl.Float32,
    "blk_addr_sim": pl.Float32,
    "blk_rank": pl.UInt16,
    "blk_rev_rank": pl.UInt16,
    "blk_flags": pl.UInt8,
}
FLAG_NAME = 1        # name TF-IDF / name similarity blocker
FLAG_ADDR = 2        # address or name+address blocker
FLAG_POSTCODE = 4    # postcode-based key blocker
FLAG_REVERSE = 8     # retrieved from the S2/S3 side (record -> best S1s)
FLAG_EXACT = 16      # exact key (e.g. identical name_core)
FLAG_KEY = 32        # exact-key routes added after pruning (src/keyroute.py)

# C3 — model scores, one file per dataset, one row per C2 row.
C3_COLUMNS = {"s1_id": pl.String, "id": pl.String, "prob": pl.Float32}

# C4 — final pairs (in memory): s1_id, id; subset of C2, each id at most once.
C4_COLUMNS = {"s1_id": pl.String, "id": pl.String}


class ContractError(ValueError):
    pass


# ---------------------------------------------------------------- paths

def _check_dataset(dataset: str) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}, expected one of {DATASETS}")


def norm_path(dataset: str, source: int) -> Path:
    _check_dataset(dataset)
    return config.CACHE_DIR / "norm" / f"{dataset}_source{source}.parquet"


def candidates_path(dataset: str) -> Path:
    _check_dataset(dataset)
    return config.CACHE_DIR / "candidates" / f"{dataset}.parquet"


def features_path(dataset: str) -> Path:
    _check_dataset(dataset)
    return config.CACHE_DIR / "features" / f"{dataset}.parquet"


def scores_path(dataset: str) -> Path:
    _check_dataset(dataset)
    return config.CACHE_DIR / "scores" / f"{dataset}.parquet"


def _meta_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


# ---------------------------------------------------------------- dataset inputs

def load_records(dataset: str, source: int, columns=None) -> pl.DataFrame:
    """Raw records (entity_id, business_name, business_address, country) of a dataset.

    `columns`: optional subset to load (the full train sources are 5M rows each)."""
    _check_dataset(dataset)
    columns = list(columns or config.SOURCE_COLUMNS)
    if dataset == "sample_5pct":
        return pl.read_parquet(config.sample_dir() / f"train_source{source}.parquet", columns=columns)
    return io.load_source(dataset, source, columns=columns)


def load_s1_folds(dataset: str) -> pl.DataFrame:
    """S1 ids of a dataset with columns (s1_id, country, fold); fold is 'test' for the test set."""
    s1 = load_records(dataset, 1, ["entity_id", "country"])
    if dataset == "test":
        s1 = s1.with_columns(pl.lit("test").alias("fold"))
    else:
        s1 = sample.with_fold(s1)
    return s1.rename({"entity_id": "s1_id"})


def load_ground_truth(dataset: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(gt_s1: s1_id, n) and (gt_pairs: s1_id, id) restricted to the dataset. Not for test."""
    _check_dataset(dataset)
    if dataset == "test":
        raise ValueError("no ground truth for the test set")
    if dataset == "sample_5pct":
        d = config.sample_dir()
        return pl.read_parquet(d / "train_gt_s1.parquet"), pl.read_parquet(d / "train_gt_pairs.parquet")
    return io.load_gt()


# ---------------------------------------------------------------- validators

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractError(msg)


def _check_schema(df: pl.DataFrame, columns: dict, name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    _require(not missing, f"{name}: missing columns {missing}")
    wrong = {c: str(df.schema[c]) for c, t in columns.items() if df.schema[c] != t}
    _require(not wrong, f"{name}: wrong dtypes {wrong}, expected {columns}")
    nulls = {c: n for c in columns if (n := df[c].null_count())}
    _require(not nulls, f"{name}: nulls in {nulls}")


def _check_pair_ids(df: pl.DataFrame, name: str) -> None:
    _require(df["s1_id"].str.starts_with("S1-").all(), f"{name}: s1_id must start with S1-")
    _require(df["id"].str.contains(r"^S[23]-").all(), f"{name}: id must start with S2- or S3-")
    # 64-bit pair hashes: exact-duplicate check without materialising 100M+ string pairs
    dup = df.height - df.select(pl.struct("s1_id", "id").hash(seed=0)).n_unique()
    _require(dup == 0, f"{name}: {dup} duplicate (s1_id, id) pairs")


def check_c1(df: pl.DataFrame, expected_ids: pl.Series | None = None) -> None:
    _check_schema(df, C1_COLUMNS, "C1")
    _require(df["entity_id"].n_unique() == df.height, "C1: duplicate entity_id")
    if expected_ids is not None:
        _require(df.height == expected_ids.len() and
                 df.join(expected_ids.to_frame("entity_id"), on="entity_id", how="anti").height == 0,
                 "C1: entity_ids differ from the dataset's records")


COUNTRY_CHECK_SAMPLE = 5_000_000


def check_c2(df: pl.DataFrame, s1: pl.DataFrame | None = None, others: pl.DataFrame | None = None) -> None:
    """`s1` / `others`: optional (entity_id, country) frames of the dataset — enables
    id-existence (on unique ids) and same-country checks (on a fixed sample of at most
    COUNTRY_CHECK_SAMPLE pairs; blocking works per country, so this guards regressions)."""
    _check_schema(df, C2_COLUMNS, "C2")
    _check_pair_ids(df, "C2")
    for c in ("blk_name_sim", "blk_addr_sim"):
        _require(df[c].is_between(0.0, 1.0001).all(), f"C2: {c} outside [0, 1]")
    _require((df["blk_rank"] >= 1).all(), "C2: blk_rank must be >= 1")
    if s1 is not None and others is not None:
        missing_s1 = df.select(pl.col("s1_id").unique().alias("entity_id")).join(s1, on="entity_id", how="anti").height
        _require(missing_s1 == 0, "C2: s1_id not in the dataset's S1 records")
        missing_id = df.select(pl.col("id").unique().alias("entity_id")).join(others, on="entity_id", how="anti").height
        _require(missing_id == 0, "C2: id not in the dataset's S2/S3 records")
        pairs = df.select("s1_id", "id")
        if pairs.height > COUNTRY_CHECK_SAMPLE:
            pairs = pairs.sample(COUNTRY_CHECK_SAMPLE, seed=0)
        j = (pairs.join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("country").alias("c1")), on="s1_id")
                  .join(others.select(pl.col("entity_id").alias("id"), pl.col("country").alias("c2")), on="id"))
        _require((j["c1"] == j["c2"]).all(), "C2: pair with different countries")


def _pair_hash(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(pl.struct("s1_id", "id").hash(seed=0).alias("h"))


def check_c3(df: pl.DataFrame, c2: pl.DataFrame) -> None:
    """Scores must cover exactly the C2 pairs of the S1 entities they contain (features may be
    built for an S1 subset, e.g. --s1-pct on full train). `c2` may be the full C2 or already
    restricted to those S1s. Pairs are compared as 64-bit hashes (test C3 is ~87M rows)."""
    _check_schema(df, C3_COLUMNS, "C3")
    _check_pair_ids(df, "C3")
    _require(df["prob"].is_between(0.0, 1.0).all(), "C3: prob outside [0, 1]")
    c2 = c2.select("s1_id", "id").join(df.select("s1_id").unique(), on="s1_id", how="semi")
    _require(df.height == c2.height and _pair_hash(df).join(_pair_hash(c2), on="h", how="anti").height == 0,
             "C3: pairs must be exactly the C2 pairs of the scored S1 entities")


def check_c4(pairs: pl.DataFrame, c2: pl.DataFrame) -> None:
    _check_schema(pairs, C4_COLUMNS, "C4")
    _check_pair_ids(pairs, "C4")
    _require(pairs.join(c2.select("s1_id", "id"), on=["s1_id", "id"], how="anti").height == 0,
             "C4: pairs must be a subset of C2")
    _require(pairs["id"].n_unique() == pairs.height, "C4: an id is assigned to more than one S1")


# ---------------------------------------------------------------- writers / loaders

def _write(df: pl.DataFrame, path: Path, producer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd")
    _meta_path(path).write_text(json.dumps(
        {"producer": producer, "rows": df.height, "written": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))


def write_norm(df: pl.DataFrame, dataset: str, source: int, producer: str) -> Path:
    check_c1(df, load_records(dataset, source, ["entity_id"])["entity_id"])
    path = norm_path(dataset, source)
    _write(df, path, producer)
    return path


def write_candidates(df: pl.DataFrame, dataset: str, producer: str) -> Path:
    cols = ["entity_id", "country"]
    s1 = load_records(dataset, 1, cols)
    others = pl.concat([load_records(dataset, s, cols) for s in (2, 3)])
    check_c2(df, s1, others)
    path = candidates_path(dataset)
    _write(df, path, producer)
    return path


def write_scores(df: pl.DataFrame, dataset: str, producer: str) -> Path:
    c2 = (pl.scan_parquet(candidates_path(dataset)).select("s1_id", "id")
          .join(df.select("s1_id").unique().lazy(), on="s1_id", how="semi").collect())
    check_c3(df, c2)
    path = scores_path(dataset)
    _write(df, path, producer)
    return path


def load_norm(dataset: str, source: int) -> pl.DataFrame:
    return pl.read_parquet(norm_path(dataset, source))


def load_candidates(dataset: str) -> pl.DataFrame:
    return pl.read_parquet(candidates_path(dataset))


def load_scores(dataset: str) -> pl.DataFrame:
    return pl.read_parquet(scores_path(dataset))


def producer_of(path: Path) -> str:
    """Who wrote a cache file (e.g. 'stub' or 'blocking v1'), from its sidecar .meta.json."""
    meta = _meta_path(path)
    return json.loads(meta.read_text())["producer"] if meta.exists() else "unknown"
