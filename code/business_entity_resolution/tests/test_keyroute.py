import polars as pl
import pytest

from src import config
from src import contracts as C
from src import keyroute as K
from src.prune import wide_path


def _norm(ids, names, addrs, hns, places):
    return pl.DataFrame({"entity_id": ids, "country": ["US"] * len(ids), "name_core": names, "addr_clean": addrs,
                         "house_no": hns, "addr_places": places})


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    (tmp_path / "norm").mkdir()
    (tmp_path / "candidates").mkdir()
    _norm(["S1-1", "S1-2"], ["behavioral health clinic", "acme"], ["132 highway 89 mayflower ar", "5 main st boston ma"],
          ["132", "5"], [["mayflower"], ["boston"]]).write_parquet(C.norm_path("test", 1))
    _norm(["S2-1", "S2-2", "S2-3", "S2-4"], ["behavioral health clinic", "zzz", "acme", "acme co"],
          ["132 highway 89 mayflower ar", "999 other rd", "", "5 main street bostn ma"], ["132", "999", "", "5"],
          [["mayflower"], ["reno"], [], ["bostn"]]).write_parquet(C.norm_path("test", 2))
    _norm(["S3-1"], ["health clinic behavioral"], ["mayflower ar 132 highway 89"], ["132"], [["mayflower"]]
          ).write_parquet(C.norm_path("test", 3))
    # S3-1 is already a (wide and pruned) candidate of S1-1
    c2 = pl.DataFrame({"s1_id": ["S1-1", "S1-2"], "id": ["S3-1", "S2-2"]}).with_columns(
        pl.lit(0.5, pl.Float32).alias("blk_name_sim"), pl.lit(0.5, pl.Float32).alias("blk_addr_sim"),
        pl.lit(1, pl.UInt16).alias("blk_rank"), pl.lit(1, pl.UInt16).alias("blk_rev_rank"), pl.lit(1, pl.UInt8).alias("blk_flags"))
    c2.write_parquet(wide_path(C.candidates_path("test")))
    c2.write_parquet(C.candidates_path("test"))
    recs = {s: pl.read_parquet(C.norm_path("test", s), columns=["entity_id", "country"]) for s in (1, 2, 3)}
    monkeypatch.setattr(C, "load_records", lambda ds, s, columns=None: recs[s])


def test_add_finds_exact_key_pairs_and_is_idempotent(cache):
    K.add("test")
    first = pl.read_parquet(C.candidates_path("test")).sort("s1_id", "id")
    key = first.filter((pl.col("blk_flags") & C.FLAG_KEY) > 0)
    # identical name+address copy, the address-less record with the identical name, and the misspelt-city
    # copy (house number + street word); not the unrelated one
    assert set(key.select("s1_id", "id").rows()) == {("S1-1", "S2-1"), ("S1-2", "S2-3"), ("S1-2", "S2-4")}
    assert first.height == 5 and first.select("s1_id", "id").is_duplicated().sum() == 0
    K.add("test")
    assert pl.read_parquet(C.candidates_path("test")).sort("s1_id", "id").equals(first)
    assert pl.read_parquet(wide_path(C.candidates_path("test"))).height == 5
