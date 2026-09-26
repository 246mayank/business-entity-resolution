import polars as pl
import pytest

from src import contracts as C
from src.stub import stub_normalize


def c2(rows):
    df = pl.DataFrame(rows, schema=["s1_id", "id"], orient="row")
    return df.with_columns(
        pl.lit(0.5, pl.Float32).alias("blk_name_sim"),
        pl.lit(0.5, pl.Float32).alias("blk_addr_sim"),
        pl.lit(1, pl.UInt16).alias("blk_rank"),
        pl.lit(0, pl.UInt16).alias("blk_rev_rank"),
        pl.lit(C.FLAG_NAME, pl.UInt8).alias("blk_flags"),
    )


CANDS = c2([("S1-1", "S2-1"), ("S1-1", "S3-1"), ("S1-2", "S2-1")])
S1 = pl.DataFrame({"entity_id": ["S1-1", "S1-2"], "country": ["US", "France"]})
OTHERS = pl.DataFrame({"entity_id": ["S2-1", "S3-1"], "country": ["US", "US"]})


def test_c2_accepts_valid_frame():
    C.check_c2(CANDS)


@pytest.mark.parametrize("bad", [
    CANDS.drop("blk_flags"),
    CANDS.with_columns(pl.col("blk_rank").cast(pl.Int64)),
    CANDS.vstack(CANDS.head(1)),
    CANDS.with_columns(pl.lit("S1-9").alias("id")),
    CANDS.with_columns(pl.lit(1.5, pl.Float32).alias("blk_name_sim")),
    CANDS.with_columns(pl.lit(None, pl.String).alias("s1_id")),
])
def test_c2_rejects_bad_frames(bad):
    with pytest.raises(C.ContractError):
        C.check_c2(bad)


def test_c2_rejects_cross_country_pair():
    with pytest.raises(C.ContractError, match="different countries"):
        C.check_c2(CANDS, S1, OTHERS)


def test_c4_enforces_one_owner_and_subset():
    ok = pl.DataFrame({"s1_id": ["S1-1", "S1-1"], "id": ["S2-1", "S3-1"]})
    C.check_c4(ok, CANDS)
    with pytest.raises(C.ContractError, match="more than one S1"):
        C.check_c4(pl.DataFrame({"s1_id": ["S1-1", "S1-2"], "id": ["S2-1", "S2-1"]}), CANDS)
    with pytest.raises(C.ContractError, match="subset"):
        C.check_c4(pl.DataFrame({"s1_id": ["S1-2"], "id": ["S3-1"]}), CANDS)


def test_c3_must_cover_c2_of_scored_s1_exactly():
    scores = CANDS.select("s1_id", "id", pl.lit(0.3, pl.Float32).alias("prob"))
    C.check_c3(scores, CANDS)
    C.check_c3(scores.filter(pl.col("s1_id") == "S1-1"), CANDS)      # S1 subset (--s1-pct) is fine
    with pytest.raises(C.ContractError):
        C.check_c3(scores.head(1), CANDS)                              # S1-1 missing one of its pairs
    with pytest.raises(C.ContractError):
        C.check_c3(scores.with_columns(pl.lit("S3-9").alias("id")).head(2).unique("id"), CANDS)


def test_stub_normalize_is_contract_valid():
    raw = pl.DataFrame({
        "entity_id": ["S2-1", "S2-2", "S2-3"],
        "business_name": ["राम मार्केटिंग प्राइवेट लिमिटेड", "-- Holloway Peak Inc Seafood", "LLC Moncada Léarning Center"],
        "business_address": ["KH NO. -570/13, NEW DELHI, 110018", "105 ELM ST, MORGANTON, NC", ""],
        "country": ["India", "US", "US"],
    })
    out = stub_normalize(raw)
    C.check_c1(out, raw["entity_id"])
    assert out["is_native_script"].to_list() == [True, False, False]
    assert out["postcode"].to_list() == ["110018", "", ""]
    assert out["house_no"][1] == "105"
    assert out["name_clean"].str.contains(r"^[a-z0-9 ]*$").all()
