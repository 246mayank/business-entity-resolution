"""Build a small, deterministic development sample of the training data.

Keeps SAMPLE_PCT% of S1 entities with all their ground-truth matches, plus
SAMPLE_PCT% of every other S2/S3 record (orphans and records owned by S1
entities outside the sample) as distractors. Each sampled S1 therefore competes
against ~12x fewer S2/S3 records than in the full data, so blocking and
precision look better on the sample than they will on the full run.

    python -m src.sample [--pct 5]
"""
import polars as pl

from . import config
from . import io_utils as io


def id_num(col: str = "entity_id") -> pl.Expr:
    return pl.col(col).str.slice(3).cast(pl.Int64)


def with_fold(s1: pl.DataFrame, col: str = "entity_id") -> pl.DataFrame:
    """Add fold column: 'valid' for ~20% of S1 entities, else 'train'."""
    return s1.with_columns(
        pl.when((id_num(col) // 100) % config.VALID_MOD == 0)
        .then(pl.lit("valid")).otherwise(pl.lit("train")).alias("fold")
    )


def build_sample(pct: int = config.SAMPLE_PCT) -> None:
    out = config.sample_dir(pct)
    out.mkdir(parents=True, exist_ok=True)

    s1 = io.load_source("train", 1).filter(id_num() % 100 < pct)
    s1 = with_fold(s1)
    gt_s1, all_pairs = io.load_gt()
    gt_s1 = gt_s1.join(s1.select(pl.col("entity_id").alias("s1_id")), on="s1_id", how="semi")
    gt_pairs = all_pairs.join(gt_s1.select("s1_id"), on="s1_id", how="semi")
    all_owned = all_pairs.select("id")

    s1.write_parquet(out / "train_source1.parquet")
    gt_s1.write_parquet(out / "train_gt_s1.parquet")
    gt_pairs.write_parquet(out / "train_gt_pairs.parquet")
    print(f"S1: {s1.height:,} entities ({gt_s1.filter(pl.col('n') == 0).height:,} singletons), "
          f"{gt_pairs.height:,} GT pairs, folds: {dict(s1['fold'].value_counts().iter_rows())}")

    for source in (2, 3):
        df = io.load_source("train", source)
        matched = df.join(gt_pairs.select(pl.col("id").alias("entity_id")), on="entity_id", how="semi")
        others = df.join(matched.select("entity_id"), on="entity_id", how="anti").filter(
            id_num() % 100 < pct
        )
        n_orphan = others.join(all_owned.rename({"id": "entity_id"}), on="entity_id", how="anti").height
        sample = pl.concat([matched, others]).sort("entity_id")
        sample.write_parquet(out / f"train_source{source}.parquet")
        print(f"S{source}: {sample.height:,} records = {matched.height:,} matched to sample "
              f"+ {others.height:,} distractors ({n_orphan:,} orphans)")


def load_sample(pct: int = config.SAMPLE_PCT) -> dict[str, pl.DataFrame]:
    d = config.sample_dir(pct)
    return {name: pl.read_parquet(d / f"{name}.parquet") for name in (
        "train_source1", "train_source2", "train_source3", "train_gt_s1", "train_gt_pairs")}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the dev sample.")
    ap.add_argument("--pct", type=int, default=config.SAMPLE_PCT)
    build_sample(ap.parse_args().pct)
