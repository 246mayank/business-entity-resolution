import numpy as np
import polars as pl
import pytest

from src import config
from src import contracts as C
from src import prune as P


def test_select_keeps_top_k_above_pmin():
    scored = pl.DataFrame({"s1_id": ["S1-1"] * 4 + ["S1-2"] * 2,
                           "id": ["S2-1", "S2-2", "S3-1", "S3-2", "S2-3", "S2-4"],
                           "p": [0.9, 0.5, 0.2, 0.005, 0.001, 0.3]})
    kept = P.select(scored, k=2, pmin=0.01).sort("s1_id", "p", descending=[False, True])
    assert kept.select("s1_id", "id").rows() == [("S1-1", "S2-1"), ("S1-1", "S2-2"), ("S1-2", "S2-4")]


class _Model:
    """Stand-in for the LightGBM pruner: probability = blk_name_sim."""
    def predict(self, X):
        return X[:, P.PRUNE_FEATURES.index("blk_name_sim")]


@pytest.fixture
def tiny_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    rng = np.random.default_rng(0)
    rows = [(f"S1-{i}", f"S{2 + j % 2}-{i}{j}") for i in range(3) for j in range(6)]
    sims = rng.uniform(0, 1, len(rows)).astype(np.float32)
    feats = pl.DataFrame({"s1_id": [r[0] for r in rows], "id": [r[1] for r in rows]}).with_columns(
        [pl.Series(c, sims if c == "blk_name_sim" else np.zeros(len(rows), np.float32)) for c in P.PRUNE_FEATURES])
    cands = feats.select("s1_id", "id").with_columns(
        pl.Series("blk_name_sim", sims), pl.lit(0.5, pl.Float32).alias("blk_addr_sim"),
        pl.lit(1, pl.UInt16).alias("blk_rank"), pl.lit(0, pl.UInt16).alias("blk_rev_rank"),
        pl.lit(1, pl.UInt8).alias("blk_flags"))
    C.features_path("test").parent.mkdir(parents=True)
    C.candidates_path("test").parent.mkdir(parents=True)
    feats.write_parquet(C.features_path("test"))
    cands.write_parquet(C.candidates_path("test"))
    recs = {1: pl.DataFrame({"entity_id": [f"S1-{i}" for i in range(3)], "country": ["US"] * 3})}
    others = pl.DataFrame({"entity_id": feats["id"], "country": ["US"] * feats.height})
    recs[2] = others.filter(pl.col("entity_id").str.starts_with("S2-"))
    recs[3] = others.filter(pl.col("entity_id").str.starts_with("S3-"))
    monkeypatch.setattr(C, "load_records", lambda ds, s, columns=None: recs[s])
    return feats


def test_apply_prunes_features_and_candidates_consistently(tiny_cache):
    keep = P.apply("test", _Model(), k=2, pmin=0.2)
    feats = pl.read_parquet(C.features_path("test"))
    cands = pl.read_parquet(C.candidates_path("test"))
    assert P.wide_path(C.features_path("test")).exists() and P.wide_path(C.candidates_path("test")).exists()
    assert set(feats.select("s1_id", "id").rows()) == set(cands.select("s1_id", "id").rows()) == set(keep.select("s1_id", "id").rows())
    assert feats.group_by("s1_id").len()["len"].max() <= 2
    assert (feats["blk_name_sim"] >= 0.2).all()
    expected = P.select(tiny_cache.select("s1_id", "id", pl.col("blk_name_sim").alias("p")), 2, 0.2)
    assert set(keep.select("s1_id", "id").rows()) == set(expected.select("s1_id", "id").rows())


def test_apply_is_repeatable_from_wide_files(tiny_cache):
    P.apply("test", _Model(), k=1, pmin=0.0)
    assert pl.read_parquet(C.features_path("test")).height == 3
    P.apply("test", _Model(), k=4, pmin=0.0)        # re-prunes from the wide files, not the pruned ones
    assert pl.read_parquet(C.features_path("test")).height == 12
    assert pl.read_parquet(P.wide_path(C.features_path("test"))).height == tiny_cache.height
