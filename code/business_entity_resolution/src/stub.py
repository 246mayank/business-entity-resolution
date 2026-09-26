"""Throwaway stand-ins for normalize.py and blocking.py so Track B can start early.

Writes contract-valid C1 and C2 for the dev sample:
    python -m src.stub
Real normalize/blocking outputs overwrite the same files later; check who wrote a
file with contracts.producer_of(path).
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from unidecode import unidecode

from . import contracts as C

DATASET = "sample_5pct"
MAX_CANDS = 50       # per S1
MAX_KEY_GROUP = 300  # skip blocking keys shared by more S2/S3 records than this
NON_LATIN = r"[ऀ-෿]"  # Indic script blocks


def _clean(col: str) -> pl.Expr:
    return (pl.col(col).str.to_lowercase()
            .str.replace_all("&", " and ")
            .str.replace_all(r"[^a-z0-9 ]", " ")
            .str.replace_all(r"\s+", " ").str.strip_chars())


def stub_normalize(df: pl.DataFrame) -> pl.DataFrame:
    ascii_expr = lambda c: pl.when(pl.col(c).str.contains(r"[^\x00-\x7F]")).then(
        pl.col(c).map_elements(unidecode, return_dtype=pl.String)).otherwise(pl.col(c))
    out = df.with_columns(
        (pl.col("business_name").str.contains(NON_LATIN) | pl.col("business_address").str.contains(NON_LATIN))
        .alias("is_native_script"),
        ascii_expr("business_name").alias("_n"),
        ascii_expr("business_address").alias("_a"),
    ).with_columns(
        _clean("_n").alias("name_clean"),
        _clean("_a").alias("addr_clean"),
    ).with_columns(
        pl.col("name_clean").alias("name_core"),
        pl.lit("").alias("legal_form"),
        pl.coalesce(pl.col("addr_clean").str.extract(r"\b(\d{6})\b"),
                    pl.col("addr_clean").str.extract(r"\b(\d{5})\b"), pl.lit("")).alias("postcode"),
        pl.col("addr_clean").str.extract(r"^(\d+[a-z]?)\b").fill_null("").alias("house_no"),
    )
    return out.select(list(C.C1_COLUMNS))


def _key_pairs(s1: pl.DataFrame, others: pl.DataFrame, key: str, flag: int) -> pl.DataFrame:
    o = others.filter(pl.col(key) != "")
    big = o.group_by("country", key).len().filter(pl.col("len") > MAX_KEY_GROUP)
    o = o.join(big, on=["country", key], how="anti")
    return (s1.filter(pl.col(key) != "")
            .join(o, on=["country", key], suffix="_o")
            .select(pl.col("entity_id").alias("s1_id"), pl.col("entity_id_o").alias("id"), pl.lit(flag, pl.UInt8).alias("f")))


def stub_candidates(n1: pl.DataFrame, n_other: pl.DataFrame) -> pl.DataFrame:
    add_key = lambda d: d.with_columns(pl.col("name_core").str.replace_all(" ", "").str.slice(0, 4).alias("prefix"))
    s1, others = add_key(n1), add_key(n_other)
    pairs = (pl.concat([_key_pairs(s1, others, "prefix", C.FLAG_NAME),
                        _key_pairs(s1, others, "postcode", C.FLAG_POSTCODE)])
             .group_by("s1_id", "id").agg(pl.col("f").bitwise_or().alias("blk_flags")))
    j = (pairs.join(n1.select(pl.col("entity_id").alias("s1_id"), pl.col("name_core").alias("n1"), pl.col("addr_clean").alias("a1")), on="s1_id")
         .join(n_other.select(pl.col("entity_id").alias("id"), pl.col("name_core").alias("n2"), pl.col("addr_clean").alias("a2")), on="id"))
    sim = lambda a, b: cpdist(j[a].to_list(), j[b].to_list(), scorer=fuzz.token_set_ratio,
                              workers=-1, dtype=np.float32) / np.float32(100)
    j = j.with_columns(pl.Series("blk_name_sim", sim("n1", "n2")), pl.Series("blk_addr_sim", sim("a1", "a2")))
    return (j.with_columns(
                pl.col("blk_name_sim").rank("ordinal", descending=True).over("s1_id").cast(pl.UInt16).alias("blk_rank"),
                pl.lit(0, pl.UInt16).alias("blk_rev_rank"))
            .filter(pl.col("blk_rank") <= MAX_CANDS)
            .select(list(C.C2_COLUMNS))
            .sort("s1_id", "blk_rank"))


def main() -> None:
    norm = {}
    for source in (1, 2, 3):
        norm[source] = stub_normalize(C.load_records(DATASET, source))
        path = C.write_norm(norm[source], DATASET, source, producer="stub")
        print(f"C1 {path.name}: {norm[source].height:,} rows")
    cands = stub_candidates(norm[1], pl.concat([norm[2], norm[3]]))
    path = C.write_candidates(cands, DATASET, producer="stub")
    print(f"C2 {path.name}: {cands.height:,} pairs, {cands.height / norm[1].height:.1f} per S1")


if __name__ == "__main__":
    main()
