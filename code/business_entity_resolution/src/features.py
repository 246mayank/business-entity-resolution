"""Feature engineering for candidate pairs (C2 → features Parquet).

Reads normalised records (C1) and candidate pairs (C2), computes similarity
features for each pair, and writes cache/features/{dataset}.parquet.

Run:
    python -m src.features [--dataset sample_5pct|train|test]
"""
import argparse
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

from . import config, contracts as C

CHUNK_SIZE = 50_000   # pairs per rapidfuzz batch
TFIDF_CHUNK = 100_000  # pairs per TF-IDF cosine batch


# ------------------------------------------------------------------ TF-IDF

def _fit_tfidf(texts: list[str]) -> TfidfVectorizer:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=3,
                          max_features=80_000, dtype=np.float32, sublinear_tf=True)
    vec.fit(texts)
    return vec


def _cosine_chunked(vec: TfidfVectorizer, a_texts: list[str],
                    b_texts: list[str]) -> np.ndarray:
    """Chunked row-wise cosine similarity to avoid OOM on large pair sets."""
    out = np.empty(len(a_texts), dtype=np.float32)
    for i in range(0, len(a_texts), TFIDF_CHUNK):
        sl = slice(i, i + TFIDF_CHUNK)
        va = vec.transform(a_texts[sl])
        vb = vec.transform(b_texts[sl])
        num = np.array(va.multiply(vb).sum(axis=1)).ravel()
        na  = np.sqrt(np.array(va.power(2).sum(axis=1))).ravel()
        nb  = np.sqrt(np.array(vb.power(2).sum(axis=1))).ravel()
        denom = na * nb
        out[sl] = np.where(denom > 0, num / denom, 0.0).astype(np.float32)
    return out


# ------------------------------------------------------------------ rapidfuzz helpers

def _fuzz_batch(a: list[str], b: list[str], scorer) -> np.ndarray:
    out = np.empty(len(a), dtype=np.float32)
    for i in range(0, len(a), CHUNK_SIZE):
        sl = slice(i, i + CHUNK_SIZE)
        out[sl] = cpdist(a[sl], b[sl], scorer=scorer, workers=-1,
                         dtype=np.float32) / np.float32(100)
    return out


# ------------------------------------------------------------------ main

def build_features(dataset: str) -> pl.DataFrame:
    t0 = time.time()

    # Load C2 candidates
    cands = C.load_candidates(dataset)

    # Load C1 norm for all three sources
    n1 = C.load_norm(dataset, 1)
    n2 = C.load_norm(dataset, 2)
    n3 = C.load_norm(dataset, 3)
    nother = pl.concat([n2, n3])

    # Join pair text fields
    pairs = (
        cands
        .join(n1.select("entity_id", "name_core", "addr_clean", "legal_form",
                        "postcode", "house_no"),
              left_on="s1_id", right_on="entity_id")
        .rename({"name_core": "nc1", "addr_clean": "ac1",
                 "legal_form": "lf1", "postcode": "pc1", "house_no": "hn1"})
        .join(nother.select("entity_id", "name_core", "addr_clean", "legal_form",
                            "postcode", "house_no"),
              left_on="id", right_on="entity_id")
        .rename({"name_core": "nc2", "addr_clean": "ac2",
                 "legal_form": "lf2", "postcode": "pc2", "house_no": "hn2"})
    )

    nc1 = pairs["nc1"].to_list()
    nc2 = pairs["nc2"].to_list()
    ac1 = pairs["ac1"].to_list()
    ac2 = pairs["ac2"].to_list()

    # --- rapidfuzz name scores
    name_tsr   = _fuzz_batch(nc1, nc2, fuzz.token_set_ratio)
    name_tsort = _fuzz_batch(nc1, nc2, fuzz.token_sort_ratio)
    name_part  = _fuzz_batch(nc1, nc2, fuzz.partial_ratio)
    name_ratio = _fuzz_batch(nc1, nc2, fuzz.ratio)

    # --- rapidfuzz address scores (empty address → 0)
    addr_tsr  = _fuzz_batch(ac1, ac2, fuzz.token_set_ratio)
    addr_part = _fuzz_batch(ac1, ac2, fuzz.partial_ratio)

    # --- TF-IDF cosine on name (fit on train S1 + others name_core)
    if dataset == "test":
        # fit on train to avoid leakage
        train_n1 = C.load_norm("train", 1)
        train_n2 = C.load_norm("train", 2)
        train_n3 = C.load_norm("train", 3)
        fit_texts = (pl.concat([train_n1, train_n2, train_n3])["name_core"].to_list())
    else:
        fit_texts = (pl.concat([n1, nother])["name_core"].to_list())

    name_vec = _fit_tfidf(fit_texts)
    name_cos = _cosine_chunked(name_vec, nc1, nc2)

    # TF-IDF cosine on address
    addr_texts_fit = (pl.concat([n1, nother])["addr_clean"].to_list()
                      if dataset != "test" else
                      pl.concat([train_n1, train_n2, train_n3])["addr_clean"].to_list())
    addr_vec = _fit_tfidf(addr_texts_fit)
    addr_cos = _cosine_chunked(addr_vec, ac1, ac2)

    # --- exact key features
    lf_match = (pairs["lf1"] == pairs["lf2"]).cast(pl.Int8)
    pc_match  = ((pairs["pc1"] != "") & (pairs["pc1"] == pairs["pc2"])).cast(pl.Int8)
    hn_match  = ((pairs["hn1"] != "") & (pairs["hn1"] == pairs["hn2"])).cast(pl.Int8)

    # --- rank gap: difference in blk_name_sim between rank-1 and this pair
    rank1_sim = (
        cands.filter(pl.col("blk_rank") == 1)
        .select("s1_id", pl.col("blk_name_sim").alias("rank1_sim"))
    )
    pairs = pairs.join(rank1_sim, on="s1_id", how="left").with_columns(
        pl.col("rank1_sim").fill_null(0.0)
    )
    rank_gap = (pairs["rank1_sim"] - pairs["blk_name_sim"]).to_numpy().astype(np.float32)

    feat = pairs.select("s1_id", "id").with_columns([
        pl.Series("name_tsr",   name_tsr),
        pl.Series("name_tsort", name_tsort),
        pl.Series("name_part",  name_part),
        pl.Series("name_ratio", name_ratio),
        pl.Series("name_cos",   name_cos),
        pl.Series("addr_tsr",   addr_tsr),
        pl.Series("addr_part",  addr_part),
        pl.Series("addr_cos",   addr_cos),
        lf_match.alias("lf_match"),
        pc_match.alias("pc_match"),
        hn_match.alias("hn_match"),
        pl.Series("rank_gap",   rank_gap),
        pairs["blk_name_sim"],
        pairs["blk_addr_sim"],
        pairs["blk_rank"].cast(pl.Float32).alias("blk_rank"),
        pairs["blk_rev_rank"].cast(pl.Float32).alias("blk_rev_rank"),
        pairs["blk_flags"].cast(pl.Float32).alias("blk_flags"),
    ])

    path = C.features_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    feat.write_parquet(path, compression="zstd")
    print(f"features {dataset}: {feat.height:,} pairs, {len(feat.columns)-2} features "
          f"in {time.time()-t0:.1f}s → {path}")
    return feat


FEATURE_COLS = [
    "name_tsr", "name_tsort", "name_part", "name_ratio", "name_cos",
    "addr_tsr", "addr_part", "addr_cos",
    "lf_match", "pc_match", "hn_match", "rank_gap",
    "blk_name_sim", "blk_addr_sim", "blk_rank", "blk_rev_rank", "blk_flags",
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sample_5pct",
                    choices=list(C.DATASETS))
    args = ap.parse_args()
    build_features(args.dataset)
