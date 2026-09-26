import numpy as np
import polars as pl
import pytest

from src import blocking as B
from src import contracts as C
from src.normalize import normalize_frame


def records(prefix, rows, country="US"):
    return pl.DataFrame({"entity_id": [f"{prefix}-{i}" for i in range(len(rows))],
                         "business_name": [r[0] for r in rows], "business_address": [r[1] for r in rows],
                         "country": [country] * len(rows)})


S1 = [("Harbor Esports LLC", "167 Kirby Drive, Clarksville, TN"),
      ("Downtown Auto Body", "55 Hamlet Street, Newton, MA"),
      ("Atlantic Center PLLC", "5286 Deer Trace, Kent, OH")]
POOL_S2 = [("HARBOR ESPORTS", "167 KIRBY DR, CLARKSVILLE, TN"),
           ("downtownautobody.com", "55 HAMLT ST, NEWTON, MA"),
           ("Zetazeph", "5286 DEER TRACE, KENT, OH")]
POOL_S3 = [("Lyraxylo Co DBA Harbor Esports L.L.C.", "Clarksville, 167 Kirby Dr, Tennessee")]
# filler so that document-frequency limits and min_df behave like real data
FILLER = [(f"Filler Business {k} Inc", f"{k} Filler Road, Springfield, IL") for k in range(40)]


@pytest.fixture(autouse=True)
def small_limits(monkeypatch):
    monkeypatch.setattr(B, "MAX_DF", 1.0)          # relative: keep every token on tiny data
    monkeypatch.setattr(B, "N_THREADS", 1)


def test_block_country_finds_true_pairs_and_is_contract_valid():
    s1 = normalize_frame(records("S1", S1 + FILLER[:10]), {})
    pool = normalize_frame(pl.concat([records("S2", POOL_S2 + FILLER[10:]), records("S3", POOL_S3)]), {})
    p = B.block_country(s1, pool)
    c2 = p.with_columns(s1["entity_id"].gather(p["i"]).alias("s1_id"), pool["entity_id"].gather(p["j"]).alias("id")
                        ).select(list(C.C2_COLUMNS))
    C.check_c2(c2)
    got = set(zip(c2["s1_id"].to_list(), c2["id"].to_list()))
    for pair in [("S1-0", "S2-0"), ("S1-0", "S3-0"), ("S1-1", "S2-1"), ("S1-2", "S2-2")]:
        assert pair in got, pair
    best = c2.filter(pl.col("blk_rank") == 1).select("s1_id", "id").rows()
    assert ("S1-0", "S2-0") in best or ("S1-0", "S3-0") in best
    assert c2.group_by("s1_id").len()["len"].max() <= B.MAX_CANDS


def test_topk_returns_best_neighbours():
    vec = B._word_vec().fit(["a b", "a c", "d e", "a b c"])
    A, M = vec.transform(["a b"]), vec.transform(["d e", "a b", "a c"])
    r, c, v = B._topk(A, M, 2)
    assert list(r) == [0, 0]
    assert c[np.argmax(v)] == 1 and v.max() == pytest.approx(1.0, abs=1e-5)


def test_rowwise_cosine_matches_dense():
    vec = B._word_vec().fit(["a b", "a c", "b c"])
    A, M = vec.transform(["a b", "b c"]), vec.transform(["a c", "b c"])
    got = B._rowwise_cos(A, M, np.array([0, 1]), np.array([0, 1]))
    dense = [(A[0] @ M[0].T).toarray()[0, 0], (A[1] @ M[1].T).toarray()[0, 0]]
    assert got == pytest.approx(dense, abs=1e-6)
