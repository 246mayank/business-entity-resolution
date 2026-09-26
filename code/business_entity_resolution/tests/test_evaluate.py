import polars as pl
import pytest

from src.evaluate import blocking_report, macro_f05, per_entity_f05


def pairs(rows):
    return pl.DataFrame(rows, schema=["s1_id", "id"], orient="row")


GT_S1 = pl.DataFrame({"s1_id": ["S1-1", "S1-2", "S1-3"], "n": [2, 1, 0]}, schema_overrides={"n": pl.UInt32})
GT = pairs([("S1-1", "S2-47"), ("S1-1", "S3-812"), ("S1-2", "S3-4")])


def test_problem_statement_example():
    pred = pairs([("S1-1", "S2-47"), ("S1-1", "S2-193"), ("S1-1", "S3-812")])
    e = per_entity_f05(pred, GT_S1, GT).filter(pl.col("s1_id") == "S1-1")
    assert e["f05"][0] == pytest.approx(0.714, abs=1e-3)


def test_perfect_prediction_scores_one():
    assert macro_f05(GT, GT_S1, GT) == pytest.approx(1.0)


def test_empty_prediction_scores_singletons_only():
    assert macro_f05(pairs([]), GT_S1, GT) == pytest.approx(1 / 3)


def test_false_positive_on_singleton_scores_zero():
    pred = GT.vstack(pairs([("S1-3", "S2-9")]))
    e = per_entity_f05(pred, GT_S1, GT).filter(pl.col("s1_id") == "S1-3")
    assert e["f05"][0] == 0.0


def test_duplicate_predictions_are_ignored():
    assert macro_f05(GT.vstack(GT), GT_S1, GT) == pytest.approx(1.0)


def test_blocking_oracle():
    cand = pairs([("S1-1", "S2-47"), ("S1-1", "S2-99"), ("S1-2", "S3-4"), ("S1-3", "S2-5")])
    rep = blocking_report(cand, GT_S1, GT)
    assert rep["pair_recall"] == pytest.approx(2 / 3)
    # oracle: S1-1 gets 1 of 2 -> 1.25/(1+0.5)=0.8333, S1-2 -> 1, S1-3 -> 1
    assert rep["oracle_macro_f05"] == pytest.approx((1.25 / 1.5 + 1 + 1) / 3)
