"""Feature engineering for candidate pairs (C2 → features Parquet).

Run:
    python -m src.features [--dataset sample_5pct|train|test]
"""
import argparse
import json
import time

import joblib
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config, contracts as C

CHUNK_SIZE  = 30_000   # pairs per rapidfuzz batch
TFIDF_CHUNK = 50_000   # pairs per TF-IDF cosine batch

TFIDF_NAME_PATH = config.MODELS_DIR / "tfidf_name.joblib"
TFIDF_ADDR_PATH = config.MODELS_DIR / "tfidf_addr.joblib"


# ------------------------------------------------------------------ TF-IDF

def _fit_tfidf(texts: list[str]) -> TfidfVectorizer:
    rng = np.random.default_rng(42)
    if len(texts) > 200_000:
        texts = list(rng.choice(texts, 200_000, replace=False))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=5,
                          max_features=50_000, dtype=np.float32, sublinear_tf=True)
    vec.fit(texts)
    return vec


def _get_tfidf_vecs(dataset: str, n1: pl.DataFrame, nother: pl.DataFrame):
    """Return (name_vec, addr_vec) fitted on train, saved/loaded from disk."""
    if TFIDF_NAME_PATH.exists() and TFIDF_ADDR_PATH.exists():
        return joblib.load(TFIDF_NAME_PATH), joblib.load(TFIDF_ADDR_PATH)

    # Fit on train C1 (or sample if train not available)
    fit_ds = "train" if C.norm_path("train", 1).exists() else dataset
    if fit_ds == dataset:
        fit_n1, fit_nother = n1, nother
    else:
        fit_n1 = C.load_norm(fit_ds, 1).select("name_core", "addr_clean")
        fit_nother = pl.concat([
            C.load_norm(fit_ds, s).select("name_core", "addr_clean") for s in (2, 3)
        ])

    name_vec = _fit_tfidf(pl.concat([fit_n1, fit_nother])["name_core"].to_list())
    addr_vec = _fit_tfidf(pl.concat([fit_n1, fit_nother])["addr_clean"].to_list())

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(name_vec, TFIDF_NAME_PATH)
    joblib.dump(addr_vec, TFIDF_ADDR_PATH)
    return name_vec, addr_vec


def _cosine_chunked(vec: TfidfVectorizer, a_texts: list[str], b_texts: list[str]) -> np.ndarray:
    out = np.empty(len(a_texts), dtype=np.float32)
    for i in range(0, len(a_texts), TFIDF_CHUNK):
        sl = slice(i, i + TFIDF_CHUNK)
        va = vec.transform(a_texts[sl])
        vb = vec.transform(b_texts[sl])
        num  = np.array(va.multiply(vb).sum(axis=1)).ravel()
        na   = np.sqrt(np.array(va.power(2).sum(axis=1))).ravel()
        nb   = np.sqrt(np.array(vb.power(2).sum(axis=1))).ravel()
        denom = na * nb
        out[sl] = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0).astype(np.float32)
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

S1_CHUNK = 20_000  # S1 entities per feature chunk (~1M pairs), bounds peak memory

NORM_COLS = ["entity_id", "name_core", "addr_clean", "legal_form",
             "house_no", "is_native_script", "addr_places", "name_alias"]

FLAG_BITS = [("flag_name", 1), ("flag_addr", 2), ("flag_char4", 4),
             ("flag_reverse", 8), ("flag_combo", 16)]


def _load_norm(ds: str, s: int) -> pl.DataFrame:
    path = C.norm_path(ds, s)
    available = pl.read_parquet_schema(path)
    return pl.read_parquet(path, columns=[c for c in NORM_COLS if c in available])


def _load_norm_ids(ds: str, ids: pl.DataFrame) -> pl.DataFrame:
    """S2+S3 C1 rows restricted to the ids that appear in the candidates."""
    ids = ids.lazy().rename({"id": "entity_id"})
    frames = []
    for s in (2, 3):
        path = C.norm_path(ds, s)
        available = pl.read_parquet_schema(path)
        frames.append(pl.scan_parquet(path)
                      .select([c for c in NORM_COLS if c in available])
                      .join(ids, on="entity_id", how="semi").collect())
    return pl.concat(frames)


def _side(df: pl.DataFrame, key: str, suffix: str, name_freq: pl.DataFrame) -> pl.DataFrame:
    mapping = {"entity_id": key, "name_core": f"nc{suffix}", "addr_clean": f"ac{suffix}",
               "legal_form": f"lf{suffix}", "house_no": f"hn{suffix}",
               "is_native_script": f"ns{suffix}", "addr_places": f"ap{suffix}",
               "name_alias": f"al{suffix}"}
    df = df.rename({k: v for k, v in mapping.items() if k in df.columns})
    return df.join(name_freq.rename({"_nfreq": f"_nfreq{suffix}"}),
                   left_on=f"nc{suffix}", right_on="name_core", how="left")


def _three_state(a: str, b: str) -> pl.Expr:
    """1 = both present and equal, 0 = one or both missing, -1 = both present and different."""
    x, y = pl.col(a).fill_null(""), pl.col(b).fill_null("")
    return (pl.when((x == "") | (y == "")).then(0)
            .when(x == y).then(1).otherwise(-1).cast(pl.Int8))


MAX_NUMS = 6  # numbers kept per address; bounds the cross product


def _loggap(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    # slice keeps long digit runs (phone-like) inside Int64
    x = a.str.slice(0, 9).cast(pl.Int64, strict=False)
    y = b.str.slice(0, 9).cast(pl.Int64, strict=False)
    return (x - y).abs().cast(pl.Float64).log1p()


def _number_features(pairs: pl.DataFrame) -> pl.DataFrame:
    """Agreement of the numbers in both addresses. -1 = no numbers / no unmatched pair.

    num_min_edit / num_min_loggap compare the closest UNMATCHED numbers: one wrong digit
    with a big gap is a typo (748 vs 548); a small gap is a neighbour (2579 vs 2582)."""
    def nums(col):
        return (pl.col(col).fill_null("").str.extract_all(r"\d+")
                .list.unique(maintain_order=True).list.head(MAX_NUMS))

    n = pairs.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias("_r"),
                     nums("ac1").alias("n1"), nums("ac2").alias("n2"))
    both = (pl.col("n1").list.len() > 0) & (pl.col("n2").list.len() > 0)
    inter = pl.col("n1").list.set_intersection(pl.col("n2")).list.len()
    union = pl.col("n1").list.set_union(pl.col("n2")).list.len()
    n = n.with_columns(both.cast(pl.Int8).alias("num_both"),
                       pl.when(both).then(inter / union).otherwise(-1.0)
                       .cast(pl.Float32).alias("num_jaccard"))

    # closest pair among the numbers that did NOT match (shared ones would give 0)
    x = (n.filter(pl.col("num_both") == 1)
          .select("_r", pl.col("n1").list.set_difference(pl.col("n2")),
                  pl.col("n2").list.set_difference(pl.col("n1")))
          .filter((pl.col("n1").list.len() > 0) & (pl.col("n2").list.len() > 0))
          .explode("n1").explode("n2"))
    x = x.with_columns(
        pl.Series("ed", cpdist(x["n1"].to_list(), x["n2"].to_list(),
                               scorer=Levenshtein.distance, workers=-1, dtype=np.int32)),
        _loggap(pl.col("n1"), pl.col("n2")).alias("gap"))
    agg = x.group_by("_r").agg(pl.col("ed").min().cast(pl.Float32).alias("num_min_edit"),
                               pl.col("gap").min().cast(pl.Float32).alias("num_min_loggap"))
    del x

    h1, h2 = pairs["hn1"].fill_null(""), pairs["hn2"].fill_null("")
    hn_both = ((h1 != "") & (h2 != "")).to_numpy()
    hn_ed = cpdist(h1.to_list(), h2.to_list(), scorer=Levenshtein.distance,
                   workers=-1, dtype=np.int32).astype(np.float32)
    hn_gap = pairs.select(_loggap(pl.col("hn1").str.extract(r"(\d+)"),
                                  pl.col("hn2").str.extract(r"(\d+)")))[:, 0].to_numpy()

    return (n.select("_r", "num_both", "num_jaccard")
             .join(agg, on="_r", how="left")
             .sort("_r")
             .with_columns(pl.col("num_min_edit", "num_min_loggap").fill_null(-1.0),
                           pl.Series("hn_edit", np.where(hn_both, hn_ed, -1.0).astype(np.float32)),
                           pl.Series("hn_loggap", np.where(hn_both & ~np.isnan(hn_gap), hn_gap, -1.0)
                                     .astype(np.float32)))
             .drop("_r"))


PEER_K = 10  # peers per S1: its top candidates by addr_name_combo
PEER_COLS = ["peer_addr_max", "peer_addr_mean", "peer_n_agree", "peer_num_max"]


def _peer_features(pairs: pl.DataFrame, combo: np.ndarray) -> pl.DataFrame:
    """Agreement of each candidate with the S1's OTHER candidates (-1 = nothing to compare).

    An S1's true matches describe the same place, so they agree with each other; a sister
    branch at a neighbouring address (the dominant false positive) disagrees with them."""
    def nums(col):
        return (pl.col(col).fill_null("").str.extract_all(r"\d+")
                .list.unique(maintain_order=True).list.head(MAX_NUMS))

    p = pairs.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias("_r"), "s1_id",
                     pl.col("ac2").fill_null("").alias("a"), nums("ac2").alias("n"),
                     pl.Series("combo", combo))
    peers = (p.filter(pl.col("combo").rank("ordinal", descending=True).over("s1_id") <= PEER_K)
              .select("s1_id", pl.col("_r").alias("_r2"), pl.col("a").alias("a2"), pl.col("n").alias("n2")))
    x = (p.select("_r", "s1_id", "a", "n").join(peers, on="s1_id")
          .filter(pl.col("_r") != pl.col("_r2")).drop("s1_id", "_r2"))

    both_a = (x["a"] != "") & (x["a2"] != "")
    sim = cpdist(x["a"].to_list(), x["a2"].to_list(), scorer=fuzz.token_set_ratio,
                 workers=-1, dtype=np.float32) / np.float32(100)
    x = x.with_columns(pl.when(pl.Series(both_a)).then(pl.Series("s", sim)).otherwise(None).alias("s"))
    both_n = (pl.col("n").list.len() > 0) & (pl.col("n2").list.len() > 0)
    x = x.with_columns(pl.when(both_n).then(
            pl.col("n").list.set_intersection(pl.col("n2")).list.len()
            / pl.col("n").list.set_union(pl.col("n2")).list.len()).otherwise(None).alias("nj"))

    agg = x.group_by("_r").agg(pl.col("s").max().alias("peer_addr_max"),
                               pl.col("s").mean().alias("peer_addr_mean"),
                               (pl.col("s") >= 0.9).sum().alias("peer_n_agree"),
                               pl.col("nj").max().alias("peer_num_max"))
    return (p.select("_r").join(agg, on="_r", how="left").sort("_r")
             .select([pl.col(c).cast(pl.Float32).fill_null(-1.0) for c in PEER_COLS]))


def _chunk_features(pairs: pl.DataFrame, name_vec, addr_vec, with_peers: bool = True) -> pl.DataFrame:
    nc1, nc2 = pairs["nc1"].fill_null("").to_list(), pairs["nc2"].fill_null("").to_list()
    ac1, ac2 = pairs["ac1"].fill_null("").to_list(), pairs["ac2"].fill_null("").to_list()

    num = {
        "name_tsr":   _fuzz_batch(nc1, nc2, fuzz.token_set_ratio),
        "name_tsort": _fuzz_batch(nc1, nc2, fuzz.token_sort_ratio),
        "name_part":  _fuzz_batch(nc1, nc2, fuzz.partial_ratio),
        "name_ratio": _fuzz_batch(nc1, nc2, fuzz.ratio),
        "addr_tsr":   _fuzz_batch(ac1, ac2, fuzz.token_set_ratio),
        "addr_part":  _fuzz_batch(ac1, ac2, fuzz.partial_ratio),
        "name_cos":   _cosine_chunked(name_vec, nc1, nc2),
        "addr_cos":   _cosine_chunked(addr_vec, ac1, ac2),
    }
    del ac1, ac2

    nc1_ns = [s.replace(" ", "") for s in nc1]
    nc2_ns = [s.replace(" ", "") for s in nc2]
    num["name_ns_ratio"] = _fuzz_batch(nc1_ns, nc2_ns, fuzz.ratio)
    num["name_ns_part"]  = _fuzz_batch(nc1_ns, nc2_ns, fuzz.partial_ratio)
    del nc1_ns, nc2_ns

    len1 = np.fromiter((len(s) for s in nc1), dtype=np.float32, count=len(nc1))
    len2 = np.fromiter((len(s) for s in nc2), dtype=np.float32, count=len(nc2))
    hi, lo = np.maximum(len1, len2), np.minimum(len1, len2)
    num["name_len_ratio"] = np.divide(lo, hi, out=np.ones_like(hi), where=hi > 0)
    del nc1, nc2

    both_addr = ((pairs["ac1"].fill_null("") != "") & (pairs["ac2"].fill_null("") != "")).to_numpy()
    num["addr_name_combo"] = np.where(both_addr, 0.4 * num["name_tsr"] + 0.6 * num["addr_tsr"],
                                      num["name_tsr"]).astype(np.float32)

    # DBA alias of the S2/S3 record vs the S1 name
    if "al2" in pairs.columns:
        al2 = pairs["al2"].fill_null("")
        a_sim = _fuzz_batch(al2.to_list(), pairs["nc1"].fill_null("").to_list(), fuzz.token_set_ratio)
        num["alias_tsr"] = np.where((al2 != "").to_numpy(), a_sim, -1.0).astype(np.float32)
    else:
        num["alias_tsr"] = np.full(pairs.height, -1.0, dtype=np.float32)

    peers = (_peer_features(pairs, num["addr_name_combo"]) if with_peers else
             pl.DataFrame({c: np.full(pairs.height, -1.0, dtype=np.float32) for c in PEER_COLS}))

    if "ap1" in pairs.columns and "ap2" in pairs.columns:
        inter = pl.col("ap1").list.set_intersection(pl.col("ap2")).list.len()
        union = pl.col("ap1").list.set_union(pl.col("ap2")).list.len()
        place_jaccard = pl.when(union > 0).then(inter / union).otherwise(0.0).fill_null(0.0)
    else:
        place_jaccard = pl.lit(0.0)

    ns2 = pl.col("ns2").cast(pl.Int8).fill_null(0) if "ns2" in pairs.columns else pl.lit(0)

    out = pairs.with_columns(
        [pl.Series(k, v) for k, v in num.items()]
        + _number_features(pairs).get_columns()
        + peers.get_columns()
        + [((pl.col("blk_flags") & bit) > 0).cast(pl.Int8).alias(name) for name, bit in FLAG_BITS]
        + [
            _three_state("lf1", "lf2").alias("lf_match"),
            _three_state("hn1", "hn2").alias("hn_match"),
            (pl.col("blk_name_sim").max().over("s1_id") - pl.col("blk_name_sim"))
                .cast(pl.Float32).alias("rank_gap"),
            pl.len().over("s1_id").cast(pl.Float32).alias("n_cands"),
            ((pl.col("ac1").fill_null("") == "") | (pl.col("ac2").fill_null("") == ""))
                .cast(pl.Int8).alias("addr_empty"),
            pl.col("id").str.starts_with("S3-").cast(pl.Int8).alias("is_s3"),
            ns2.cast(pl.Int8).alias("ns2"),
            place_jaccard.cast(pl.Float32).alias("place_jaccard"),
            pl.col("_nfreq1").fill_null(0).log1p().cast(pl.Float32).alias("name_freq_s1"),
            pl.col("_nfreq2").fill_null(0).log1p().cast(pl.Float32).alias("name_freq_id"),
            pl.col("blk_rank").cast(pl.Float32),
            pl.col("blk_rev_rank").cast(pl.Float32),
        ]
    )
    # within-S1 standing: is this the S1's best address / number match, and is it a
    # same-name candidate whose numbers conflict (a sister branch)?
    same_name_diff_num = ((pl.col("name_ratio") >= 0.95) & (pl.col("num_both") == 1)
                          & (pl.col("num_jaccard") == 0))
    out = out.with_columns(
        pl.col("addr_tsr").rank("min", descending=True).over("s1_id").cast(pl.Float32).alias("addr_rank_s1"),
        pl.col("num_jaccard").rank("min", descending=True).over("s1_id").cast(pl.Float32).alias("num_rank_s1"),
        same_name_diff_num.cast(pl.Int8).alias("same_name_diff_num"),
        same_name_diff_num.sum().over("s1_id").cast(pl.Float32).alias("s1_same_name_diff_num"),
    )
    return out.select(["s1_id", "id"] + [c for c in FEATURE_COLS if c not in ID_SCOPED])


# Per-id context, computed over the FULL C2 so an S1 subsample doesn't shrink the counts.
ID_SCOPED = ["n_s1_for_id", "n_same_name_cands"]
NEAR_SAME_NAME = 0.9  # blk_name_sim cutoff; agrees ~94% with name_ratio >= 0.95


def subsample_path(dataset: str):
    return C.features_path(dataset).with_suffix(".subsample.json")


def in_subsample(s1_pct: int | None, col: str = "s1_id") -> pl.Expr:
    """Deterministic S1 subsample on the last two id digits (independent of the fold hash)."""
    if s1_pct is None:
        return pl.lit(True)
    return pl.col(col).str.slice(3).cast(pl.Int64) % 100 < s1_pct


def featurized_s1_pct(dataset: str) -> int | None:
    p = subsample_path(dataset)
    return json.loads(p.read_text())["s1_pct"] if p.exists() else None


SHARD_S1 = 200_000  # S1 entities whose candidates + S2/S3 records are in memory at once


def build_features(dataset: str, s1_pct: int | None = None):
    """Features for C2 (optionally only `s1_pct`% of S1), written to C.features_path(dataset).

    Streams: candidates and S2/S3 records are loaded per shard of SHARD_S1 S1 entities, each
    S1_CHUNK chunk is written to a part file, and the parts are joined with the per-id context
    and sunk into one Parquet file, so memory does not grow with the dataset (test C2 is ~87M
    pairs). Returns the output path."""
    import shutil

    t0 = time.time()
    cand_path = C.candidates_path(dataset)
    scan = pl.scan_parquet(cand_path).filter(in_subsample(s1_pct))
    s1_ids = scan.select("s1_id").unique().collect().sort("s1_id")
    # Count competitors over the unpruned candidates when they exist. After --s1-pct + prune, the
    # pruned train file holds only the featurised S1s, while test holds all of them: counting there
    # would give test ~5x the competitors the model saw in training.
    wide = cand_path.with_name(cand_path.stem + ".wide.parquet")
    ctx_src = wide if wide.exists() else cand_path
    id_ctx = (pl.scan_parquet(ctx_src).group_by("id").agg(
                  pl.len().cast(pl.Float32).alias("n_s1_for_id"),
                  (pl.col("blk_name_sim") >= NEAR_SAME_NAME).sum().cast(pl.Float32)
                    .alias("n_same_name_cands"))
              .collect(engine="streaming"))
    print(f"  {s1_ids.height:,} S1 with candidates (s1_pct={s1_pct})", flush=True)

    n1 = _load_norm(dataset, 1)
    name_vec, addr_vec = _get_tfidf_vecs(
        dataset, n1, None if TFIDF_NAME_PATH.exists() and TFIDF_ADDR_PATH.exists()
        else _load_norm_ids(dataset, scan.select("id").unique().collect()))

    # name frequency among this dataset's S1: common names are weak evidence
    name_freq = n1.group_by("name_core").agg(pl.len().alias("_nfreq"))

    parts_dir = C.features_path(dataset).with_suffix(".parts")
    shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir(parents=True)
    n_rows, part = 0, 0
    n_shards = (s1_ids.height + SHARD_S1 - 1) // SHARD_S1
    for sh in range(n_shards):
        shard_ids = s1_ids.slice(sh * SHARD_S1, SHARD_S1)
        cands = scan.join(shard_ids.lazy(), on="s1_id", how="semi").collect()
        left = _side(n1.join(shard_ids.rename({"s1_id": "entity_id"}), on="entity_id", how="semi"),
                     "s1_id", "1", name_freq)
        right = _side(_load_norm_ids(dataset, cands.select("id").unique()), "id", "2", name_freq)
        cands = cands.with_columns(
            ((pl.col("s1_id").rank("dense") - 1) // S1_CHUNK).cast(pl.UInt32).alias("_chunk"))
        for k in range(cands["_chunk"].max() + 1):
            pairs = (cands.filter(pl.col("_chunk") == k).drop("_chunk")
                     .join(left, on="s1_id").join(right, on="id"))
            f = _chunk_features(pairs, name_vec, addr_vec, with_peers=wide.exists())
            f.write_parquet(parts_dir / f"part_{part:05d}.parquet")
            n_rows += f.height
            part += 1
            del pairs, f
        del cands, left, right
        print(f"  shard {sh + 1}/{n_shards}: {n_rows:,} pairs ({time.time() - t0:.0f}s)", flush=True)
    del n1

    path = C.features_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    (pl.scan_parquet(parts_dir / "*.parquet")
       .join(id_ctx.lazy(), on="id", how="left")
       .with_columns(pl.col(ID_SCOPED).fill_null(0))
       .select(["s1_id", "id"] + FEATURE_COLS)
       .sink_parquet(path, compression="zstd"))
    shutil.rmtree(parts_dir, ignore_errors=True)
    if s1_pct is None:
        subsample_path(dataset).unlink(missing_ok=True)
    else:
        subsample_path(dataset).write_text(json.dumps({"s1_pct": s1_pct}))
    print(f"features {dataset}: {n_rows:,} pairs, {len(FEATURE_COLS)} features "
          f"in {time.time()-t0:.1f}s → {path}")
    return path


FEATURE_COLS = [
    "name_tsr", "name_tsort", "name_part", "name_ratio",
    "name_ns_ratio", "name_ns_part", "name_cos",
    "addr_tsr", "addr_part", "addr_cos",
    "lf_match", "hn_match",
    "flag_name", "flag_addr", "flag_char4", "flag_reverse", "flag_combo",
    "rank_gap", "n_cands", "n_s1_for_id",
    "addr_empty", "name_len_ratio", "addr_name_combo",
    "is_s3", "ns2", "place_jaccard",
    "blk_name_sim", "blk_addr_sim", "blk_rank", "blk_rev_rank",
    "name_freq_s1", "name_freq_id", "n_same_name_cands",
    "num_both", "num_jaccard", "num_min_edit", "num_min_loggap", "hn_edit", "hn_loggap",
    "peer_addr_max", "peer_addr_mean", "peer_n_agree", "peer_num_max",
    "addr_rank_s1", "num_rank_s1", "same_name_diff_num", "s1_same_name_diff_num", "alias_tsr",
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sample_5pct", choices=list(C.DATASETS))
    ap.add_argument("--s1-pct", type=int, default=None,
                    help="featurize only this %% of S1 entities (for full train; never for test)")
    args = ap.parse_args()
    if args.dataset == "test" and args.s1_pct is not None:
        ap.error("--s1-pct is not allowed for test")
    build_features(args.dataset, args.s1_pct)
