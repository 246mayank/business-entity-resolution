import json
import sys
import types

import polars as pl
import pytest

from src import contracts as C
from src import io_utils as io
from src import run_pipeline as P

S1 = pl.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"], "business_name": ["a", "b", "c"],
                   "business_address": ["", "", ""], "country": ["US", "US", "France"]})
CANDS = pl.DataFrame({"s1_id": ["S1-1", "S1-1", "S1-2", "S1-2"], "id": ["S2-1", "S3-1", "S2-1", "S3-2"]})
SCORES = CANDS.with_columns(pl.Series("prob", [0.9, 0.7, 0.8, 0.2], dtype=pl.Float32))


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    monkeypatch.setattr(P.config, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(P, "PIPE_DIR", tmp_path / "pipeline")
    monkeypatch.setattr(C, "load_records", lambda dataset, source, columns=None: S1.select(columns or S1.columns))
    monkeypatch.setattr(P, "_load_cands", lambda dataset, s1: CANDS.join(s1, on="s1_id", how="semi"))
    monkeypatch.setattr(C, "load_ground_truth", lambda dataset: (
        pl.DataFrame({"s1_id": ["S1-1", "S1-2", "S1-3"], "n": [2, 0, 0]}, schema_overrides={"n": pl.UInt32}),
        pl.DataFrame({"s1_id": ["S1-1", "S1-1"], "id": ["S2-1", "S3-1"]})))
    monkeypatch.setattr(C, "load_s1_folds", lambda dataset: pl.DataFrame(
        {"s1_id": ["S1-1", "S1-2", "S1-3"], "country": ["US", "US", "France"], "fold": ["valid"] * 3}))
    decision = {"path": tmp_path / "decision.json"}
    fake = types.ModuleType("src.predict")
    fake.score = lambda dataset, model=None: SCORES
    fake.decide_pairs = lambda scores, t: (scores.filter(pl.col("prob") >= t).sort("prob", descending=True)
                                           .unique(subset=["id"], keep="first").select("s1_id", "id"))
    fake.load_decision = lambda: json.loads(decision["path"].read_text())
    monkeypatch.setitem(sys.modules, "src.predict", fake)
    feats = types.ModuleType("src.features")
    feats.state = {"pct": None}
    feats.featurized_s1_pct = lambda dataset: feats.state["pct"]
    feats.in_subsample = lambda pct, col="s1_id": (pl.lit(True) if pct is None
                                                   else pl.col(col).str.slice(3).cast(pl.Int64) % 100 < pct)
    monkeypatch.setitem(sys.modules, "src.features", feats)
    return tmp_path


def test_predict_applies_threshold_and_one_owner(fake_env):
    P.step_predict("sample_5pct", "sample_5pct", threshold=0.5)
    pairs = pl.read_parquet(P.final_pairs_path("sample_5pct")).sort("id")
    # S2-1 is wanted by S1-1 (0.9) and S1-2 (0.8): one owner -> S1-1; S3-2 is below threshold
    assert pairs.rows() == [("S1-1", "S2-1"), ("S1-1", "S3-1")]


def test_predict_reads_saved_decision(fake_env):
    (fake_env / "decision.json").write_text(json.dumps({"threshold": 0.75, "tuned_on": "train"}))
    P.step_predict("test", "train", threshold=None)
    assert pl.read_parquet(P.final_pairs_path("test")).rows() == [("S1-1", "S2-1")]


def test_predict_refuses_decision_tuned_on_another_dataset(fake_env):
    (fake_env / "decision.json").write_text(json.dumps({"threshold": 0.75, "tuned_on": "sample_5pct"}))
    with pytest.raises(SystemExit, match="tuned on 'sample_5pct'"):
        P.step_predict("test", "train", threshold=None)


def test_predict_without_decision_fails_clearly(fake_env):
    with pytest.raises(SystemExit, match="no models/decision.json"):
        P.step_predict("test", "train", threshold=None)


def test_write_restricts_to_featurized_subsample(fake_env):
    sys.modules["src.features"].state["pct"] = 2          # S1-1 only (1 % 100 < 2)
    P.step_predict("sample_5pct", "sample_5pct", threshold=0.5)
    P.step_write("sample_5pct")
    m_s1, _ = io.read_id_list_file(fake_env / "output" / "dev" / "sample_5pct" / "matching_results.tsv", io.MATCH_COL)
    assert m_s1["s1_id"].to_list() == ["S1-1"]


def test_s1_pct_not_allowed_for_test():
    with pytest.raises(SystemExit, match="not allowed for test"):
        P.step_features("test", 20)


def test_write_covers_every_s1_and_keeps_dev_outputs_separate(fake_env):
    P.step_predict("sample_5pct", "sample_5pct", threshold=0.5)
    P.step_write("sample_5pct")
    out = fake_env / "output" / "dev" / "sample_5pct"
    m_s1, m_pairs = io.read_id_list_file(out / "matching_results.tsv", io.MATCH_COL)
    c_s1, c_pairs = io.read_id_list_file(out / "candidate_pairs.tsv", io.CAND_COL)
    assert m_s1["s1_id"].to_list() == c_s1["s1_id"].to_list() == ["S1-1", "S1-2", "S1-3"]
    assert m_pairs.join(c_pairs, on=["s1_id", "id"], how="anti").height == 0
    assert not (fake_env / "output" / "matching_results.tsv").exists()


def test_test_set_writes_submission_paths(fake_env):
    P.step_predict("test", "sample_5pct", threshold=0.5)
    P.step_write("test")
    assert (fake_env / "output" / "matching_results.tsv").exists()
    assert (fake_env / "output" / "candidate_pairs.tsv").exists()


def test_test_run_requires_model_from():
    with pytest.raises(SystemExit, match="model-from"):
        P.run("test", steps=["predict"])
