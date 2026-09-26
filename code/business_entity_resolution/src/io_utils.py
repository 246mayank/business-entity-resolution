"""Reading the raw TSVs, caching them as Parquet, and writing submission files.

Run once after unzipping the data:
    python -m src.io_utils            (from code/business_entity_resolution/)
"""
import time

import polars as pl

from . import config

S1_COL = "source1_entity_id"
MATCH_COL = "matched_entity_ids"
CAND_COL = "candidate_entity_ids"


def read_tsv(path) -> pl.DataFrame:
    """Read a challenge TSV with every column as a string.

    No quote handling (names contain stray quotes), no NA inference (a business
    called "NA" or an empty address must stay a string).
    """
    return pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        infer_schema=False,
        empty_string_is_null=False,
        encoding="utf8",
    )


def convert_all(force: bool = False) -> None:
    """Convert every source TSV plus the ground truth to Parquet in cache/."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for split in config.SPLITS:
        for source in config.SOURCES:
            src, dst = config.raw_path(split, source), config.parquet_path(split, source)
            if dst.exists() and not force:
                print(f"skip {dst.name} (exists)")
                continue
            t = time.time()
            df = read_tsv(src)
            _check_source(df, split, source)
            df.write_parquet(dst, compression="zstd")
            print(f"{dst.name}: {df.height:,} rows in {time.time() - t:.1f}s")

    if config.gt_pairs_path().exists() and not force:
        print("skip ground truth (exists)")
        return
    t = time.time()
    gt = read_tsv(config.raw_gt_path())
    gt_s1, gt_pairs = parse_id_lists(gt, MATCH_COL)
    gt_s1.write_parquet(config.gt_s1_path(), compression="zstd")
    gt_pairs.write_parquet(config.gt_pairs_path(), compression="zstd")
    print(f"ground truth: {gt_s1.height:,} S1 entities, {gt_pairs.height:,} pairs "
          f"in {time.time() - t:.1f}s")


def _check_source(df: pl.DataFrame, split: str, source: int) -> None:
    assert tuple(df.columns) == config.SOURCE_COLUMNS, df.columns
    prefix = f"S{source}-"
    bad = df.filter(~pl.col("entity_id").str.starts_with(prefix)).height
    assert bad == 0, f"{split} source{source}: {bad} ids without prefix {prefix}"
    dup = df.height - df["entity_id"].n_unique()
    assert dup == 0, f"{split} source{source}: {dup} duplicate entity_ids"


def parse_id_lists(df: pl.DataFrame, list_col: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a (source1_entity_id, comma-list) table into
    per-S1 counts (s1_id, n) and long pairs (s1_id, id)."""
    df = df.rename({S1_COL: "s1_id"})
    pairs = (
        df.select("s1_id", pl.col(list_col).str.split(",").alias("id"))
        .explode("id", empty_as_null=True)
        .with_columns(pl.col("id").str.strip_chars())
        .filter(pl.col("id").is_not_null() & (pl.col("id") != ""))
        .unique(maintain_order=True)
    )
    counts = pairs.group_by("s1_id").len("n")
    s1 = (
        df.select("s1_id")
        .join(counts, on="s1_id", how="left")
        .with_columns(pl.col("n").fill_null(0).cast(pl.UInt32))
    )
    return s1, pairs


def load_source(split: str, source: int, columns=None) -> pl.DataFrame:
    return pl.read_parquet(config.parquet_path(split, source), columns=columns)


def load_gt() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (gt_s1: s1_id, n) and (gt_pairs: s1_id, id)."""
    return pl.read_parquet(config.gt_s1_path()), pl.read_parquet(config.gt_pairs_path())


def read_id_list_file(path, list_col: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read a matching_results.tsv / candidate_pairs.tsv into (s1, pairs)."""
    return parse_id_lists(read_tsv(path), list_col)


def write_id_lists(pairs: pl.DataFrame, s1_ids: pl.Series, path, list_col: str) -> None:
    """Write one row per S1 id (in the order of `s1_ids`) with a comma-joined id list.

    `pairs` has columns (s1_id, id). S1 ids with no pairs get an empty list;
    duplicate pairs are dropped; pairs for unknown S1 ids raise.
    """
    pairs = pairs.select("s1_id", "id").unique()
    unknown = pairs.join(s1_ids.to_frame("s1_id"), on="s1_id", how="anti").height
    if unknown:
        raise ValueError(f"{unknown} pairs reference S1 ids outside the given set")
    lists = pairs.sort("s1_id", "id").group_by("s1_id", maintain_order=True).agg(
        pl.col("id").str.join(",").alias(list_col)
    )
    out = (
        s1_ids.to_frame("s1_id")
        .join(lists, on="s1_id", how="left", maintain_order="left")
        .with_columns(pl.col(list_col).fill_null(""))
        .rename({"s1_id": S1_COL})
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never", line_terminator="\n")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Convert raw TSVs to Parquet cache.")
    ap.add_argument("--force", action="store_true", help="overwrite existing Parquet files")
    convert_all(force=ap.parse_args().force)
