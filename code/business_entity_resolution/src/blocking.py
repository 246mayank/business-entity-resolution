"""Candidate generation: C1 -> C2 (see contracts.py and docs/method/blocking.md).

    python -m src.blocking --dataset sample_5pct      # dev loop, prints blocking_report on the valid fold
    python -m src.blocking --dataset test

Per country (matches never cross countries), S2 and S3 records form one target pool.
Blockers, each a sparse TF-IDF top-K search (sparse_dot_topn) over the pool:
  FLAG_NAME     name_core words (1-2 grams)             top K_NAME per source
  FLAG_ADDR     addr_clean words                        top K_ADDR per source
  FLAG_EXACT    name + address combined vector          top K_COMBO per source
  FLAG_EXACT    same, but searched within each address place (records sharing a
                city/district/state component) — separates generic names by location
  FLAG_POSTCODE char 4-grams of name_nospace, only against single-token / domain names
                (squashed names like `downtownautobody.com`); the flag name is historical
  FLAG_REVERSE  every S2/S3 record -> its top K_REVERSE S1s on the combined vector
Pairs are merged, scored with name/address cosine, ranked, and capped at MAX_CANDS per S1.
"""
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from . import config
from . import contracts as C

PRODUCER = "blocking v1"

K_NAME = 10
K_ADDR = 10
K_COMBO = 10
K_CHAR = 5
K_PLACE = 10
K_REVERSE = 3
MAX_CANDS = 60
MAX_DF = 3000           # absolute: tokens in > 3000 records are dropped, so matmul cost does not grow with data size
MAX_PLACE_POOL = 300_000  # skip place groups larger than this (state-level groups are covered by the global search)
CHUNK = 50_000          # S1 rows per sparse matmul call
N_THREADS = 8


def _vectorizer(**kw) -> TfidfVectorizer:
    return TfidfVectorizer(min_df=2, max_df=MAX_DF, sublinear_tf=True, dtype=np.float32, **kw)


def _word_vec(ngrams=(1, 1)) -> TfidfVectorizer:
    return _vectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=ngrams)


def _topk(A: sp.csr_matrix, B: sp.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k cosine neighbours in B for every row of A: (row, col, score)."""
    Bt = B.T.tocsr()
    rows, cols, vals = [], [], []
    for start in range(0, A.shape[0], CHUNK):
        R = sp_matmul_topn(A[start:start + CHUNK], Bt, top_n=k, n_threads=N_THREADS).tocoo()
        rows.append(R.row + start); cols.append(R.col); vals.append(R.data)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals).astype(np.float32)


def _rowwise_cos(A: sp.csr_matrix, B: sp.csr_matrix, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """cosine(A[i[k]], B[j[k]]) for every k (rows are L2-normalised by TfidfVectorizer)."""
    out = np.empty(len(i), dtype=np.float32)
    step = 2_000_000
    for s in range(0, len(i), step):
        out[s:s + step] = np.asarray(A[i[s:s + step]].multiply(B[j[s:s + step]]).sum(axis=1)).ravel()
    return np.clip(out, 0.0, 1.0)


def _pairs(rows, cols, flag, s1_idx: np.ndarray, t_idx: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({"i": s1_idx[rows].astype(np.int64), "j": t_idx[cols].astype(np.int64),
                         "f": np.full(len(rows), flag, dtype=np.uint8)})


def _place_pairs(s1: pl.DataFrame, pool: pl.DataFrame, X1: sp.csr_matrix, XT: sp.csr_matrix) -> pl.DataFrame:
    explode = lambda df, idx: (df.select(pl.Series(idx, np.arange(df.height)), pl.col("addr_places").alias("p"))
                               .explode("p", empty_as_null=True).drop_nulls().unique())
    a, b = explode(s1, "i"), explode(pool, "j")
    sizes = (a.group_by("p").len("na").join(b.group_by("p").len("nb"), on="p")
             .filter(pl.col("nb") <= MAX_PLACE_POOL))
    a = a.join(sizes, on="p", how="semi").sort("p")
    b = b.join(sizes, on="p", how="semi").sort("p")
    ga = dict(a.group_by("p").agg("i").iter_rows())
    out = []
    for place, j_idx in b.group_by("p").agg("j").iter_rows():
        i_idx = np.asarray(ga[place]); j_idx = np.asarray(j_idx)
        r, c, _ = _topk(X1[i_idx], XT[j_idx], K_PLACE)
        out.append(_pairs(r, c, C.FLAG_EXACT, i_idx, j_idx))
    return pl.concat(out) if out else pl.DataFrame(schema={"i": pl.Int64, "j": pl.Int64, "f": pl.UInt8})


def block_country(s1: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    """Candidates for one country. s1 / pool are C1 frames; returns (i, j, blk_*) with row indices."""
    names = pl.concat([s1["name_core"], pool["name_core"]]).to_list()
    addrs = pl.concat([s1["addr_clean"], pool["addr_clean"]]).to_list()
    vn, va = _word_vec((1, 2)).fit(names), _word_vec().fit(addrs)
    N1, NT = vn.transform(s1["name_core"].to_list()), vn.transform(pool["name_core"].to_list())
    A1, AT = va.transform(s1["addr_clean"].to_list()), va.transform(pool["addr_clean"].to_list())
    # combined vector: cosine = mean of name and address cosine
    X1 = sp.hstack([N1, A1], format="csr") * np.float32(np.sqrt(0.5))
    XT = sp.hstack([NT, AT], format="csr") * np.float32(np.sqrt(0.5))

    s1_all = np.arange(s1.height)
    parts = []
    for src in ("S2-", "S3-"):
        t_idx = np.flatnonzero(pool["entity_id"].str.starts_with(src).to_numpy())
        if len(t_idx) == 0:
            continue
        for flag, Q, T, k in ((C.FLAG_NAME, N1, NT, K_NAME), (C.FLAG_ADDR, A1, AT, K_ADDR), (C.FLAG_EXACT, X1, XT, K_COMBO)):
            r, c, _ = _topk(Q, T[t_idx], k)
            parts.append(_pairs(r, c, flag, s1_all, t_idx))

    # place-conditioned combined search: within every address place shared by S1 and pool
    parts.append(_place_pairs(s1, pool, X1, XT))

    # squashed / single-token names: char 4-grams against that subset of the pool only
    hard = np.flatnonzero((pool["is_domain"] | ~pool["name_core"].str.contains(" ")).to_numpy())
    if len(hard):
        vc = _vectorizer(analyzer="char", ngram_range=(4, 4)).fit(
            pl.concat([s1["name_nospace"], pool["name_nospace"][hard]]).to_list())
        r, c, _ = _topk(vc.transform(s1["name_nospace"].to_list()), vc.transform(pool["name_nospace"][hard].to_list()), K_CHAR)
        parts.append(_pairs(r, c, C.FLAG_POSTCODE, s1_all, hard))

    # reverse: each S2/S3 record -> its best S1s
    r, c, _ = _topk(XT, X1, K_REVERSE)
    parts.append(_pairs(c, r, C.FLAG_REVERSE, s1_all, np.arange(pool.height)))

    pairs = pl.concat(parts).group_by("i", "j").agg(pl.col("f").bitwise_or().alias("blk_flags"))
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    pairs = pairs.with_columns(
        pl.Series("blk_name_sim", _rowwise_cos(N1, NT, i, j)),
        pl.Series("blk_addr_sim", _rowwise_cos(A1, AT, i, j)),
    ).with_columns(((pl.col("blk_name_sim") + pl.col("blk_addr_sim")) / 2).alias("_score"))
    pairs = pairs.with_columns(
        pl.col("_score").rank("ordinal", descending=True).over("i").alias("_rank"),
        pl.col("_score").rank("ordinal", descending=True).over("j").alias("_rev"),
    )
    # cap per S1, but never drop a record's best S1 (records have exactly one owner)
    pairs = pairs.filter((pl.col("_rank") <= MAX_CANDS) | ((pl.col("_rev") == 1) & (pl.col("_rank") <= 2 * MAX_CANDS)))
    return pairs.with_columns(
        pl.col("_score").rank("ordinal", descending=True).over("i").cast(pl.UInt16).alias("blk_rank"),
        pl.col("_score").rank("ordinal", descending=True).over("j").cast(pl.UInt16).alias("blk_rev_rank"),
    ).drop("_score", "_rank", "_rev")


def build_candidates(dataset: str) -> pl.DataFrame:
    s1_all = C.load_norm(dataset, 1)
    pool_all = pl.concat([C.load_norm(dataset, 2), C.load_norm(dataset, 3)])
    out = []
    for country in s1_all["country"].unique().sort().to_list():
        t = time.time()
        s1 = s1_all.filter(pl.col("country") == country)
        pool = pool_all.filter(pl.col("country") == country)
        if pool.height == 0:
            continue
        p = block_country(s1, pool)
        out.append(p.with_columns(
            s1["entity_id"].gather(p["i"]).alias("s1_id"),
            pool["entity_id"].gather(p["j"]).alias("id"),
        ).select(list(C.C2_COLUMNS)))
        print(f"  {country}: S1 {s1.height:,} x pool {pool.height:,} -> {p.height:,} pairs "
              f"({p.height / s1.height:.1f}/S1) in {time.time() - t:.0f}s", flush=True)
    cands = pl.concat(out).sort("s1_id", "blk_rank")
    C.write_candidates(cands, dataset, producer=PRODUCER)
    return cands


if __name__ == "__main__":
    import argparse

    from .evaluate import blocking_report, print_report

    ap = argparse.ArgumentParser(description="Build candidate pairs (C2).")
    ap.add_argument("--dataset", default="sample_5pct", choices=C.DATASETS)
    args = ap.parse_args()
    t = time.time()
    cands = build_candidates(args.dataset)
    print(f"{args.dataset}: {cands.height:,} pairs in {time.time() - t:.0f}s")
    if args.dataset != "test":
        gt_s1, gt = C.load_ground_truth(args.dataset)
        folds = C.load_s1_folds(args.dataset)
        valid = gt_s1.join(folds.filter(pl.col("fold") == "valid").select("s1_id"), on="s1_id", how="semi")
        print_report(blocking_report(cands, valid, gt), f"{args.dataset} valid fold")
        for flag, name in ((C.FLAG_NAME, "name"), (C.FLAG_ADDR, "addr"), (C.FLAG_EXACT, "combo"),
                           (C.FLAG_POSTCODE, "char4"), (C.FLAG_REVERSE, "reverse")):
            sub = cands.filter((pl.col("blk_flags") & flag) > 0)
            only = cands.filter(pl.col("blk_flags") == flag)
            r = blocking_report(sub, valid, gt)["pair_recall"]
            r_only = only.join(gt, on=["s1_id", "id"], how="semi").join(valid, on="s1_id", how="semi").height
            print(f"  {name:8s} recall alone {r:.3f}  unique true pairs {r_only:,}  pairs {sub.height:,}")
