"""Score candidates, tune threshold for macro F0.5, write decision config.

Run:
    python -m src.predict [--dataset sample_5pct|train|test] [--threshold 0.5]
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config, contracts as C, evaluate
from .features import FEATURE_COLS, featurized_s1_pct, in_subsample
from .train import load_model

DECISION_PATH = config.MODELS_DIR / "decision.json"
MIN_PRECISION = 0.987  # baseline micro precision on sample valid fold


# ------------------------------------------------------------------ scoring

def score(dataset: str, model: lgb.Booster | None = None) -> pl.DataFrame:
    if model is None:
        model = load_model()
    feat = pl.read_parquet(C.features_path(dataset))
    X = feat.select(FEATURE_COLS).to_numpy().astype(np.float32)
    probs = model.predict(X).astype(np.float32)
    scores = feat.select("s1_id", "id").with_columns(pl.Series("prob", probs))
    C.write_scores(scores, dataset, producer="predict.score")
    return scores


# ------------------------------------------------------------------ one-owner assignment

def one_owner(pairs: pl.DataFrame) -> pl.DataFrame:
    """Each S2/S3 id goes to at most one S1 (highest prob wins)."""
    return (pairs.sort("prob", descending=True)
            .unique(subset=["id"], keep="first")
            .select("s1_id", "id"))


# ------------------------------------------------------------------ pure decision

def decide_pairs(scores: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Filter by threshold, apply one-owner, validate C4 contract."""
    pairs = one_owner(scores.filter(pl.col("prob") >= threshold))
    C.check_c4(pairs, scores)
    return pairs


# ------------------------------------------------------------------ threshold tuning

def _eval_s1(dataset: str, folds: pl.DataFrame) -> pl.DataFrame:
    """Valid-fold S1s that were featurized (all of them unless features used --s1-pct)."""
    return (folds.filter((pl.col("fold") == "valid") & in_subsample(featurized_s1_pct(dataset)))
            .select("s1_id"))


def tune_threshold(dataset: str, scores: pl.DataFrame,
                   min_precision: float = MIN_PRECISION) -> float:
    gt_s1, gt_pairs = C.load_ground_truth(dataset)
    folds = C.load_s1_folds(dataset)
    s1_eval = _eval_s1(dataset, folds)
    scores_eval  = scores.join(s1_eval, on="s1_id", how="semi")
    gt_s1_eval   = gt_s1.join(s1_eval, on="s1_id", how="semi")
    gt_pairs_eval = gt_pairs.join(s1_eval, on="s1_id", how="semi")
    gt_keys = gt_pairs_eval.select("s1_id", "id")

    best_t, best_f = None, 0.0
    free_t, free_f = 0.5, 0.0
    for t in np.arange(0.05, 0.99, 0.01):
        pairs = decide_pairs(scores_eval, float(t))
        f = evaluate.macro_f05(pairs, gt_s1_eval, gt_pairs_eval)
        tp = pairs.join(gt_keys, on=["s1_id", "id"], how="semi").height
        prec = tp / pairs.height if pairs.height else 0.0
        if f > free_f:
            free_f, free_t = f, float(t)
        if prec >= min_precision and f > best_f:
            best_f, best_t = f, float(t)

    if best_t is None:
        print(f"no threshold reaches precision >= {min_precision:.3f}; "
              f"falling back to unguarded best")
        best_t, best_f = free_t, free_f
    print(f"unguarded best: t={free_t:.2f} F0.5={free_f:.4f}")
    print(f"chosen:         t={best_t:.2f} F0.5={best_f:.4f} (precision >= {min_precision:.3f})")
    return best_t


# ------------------------------------------------------------------ decide (with reporting)

def decide(dataset: str, threshold: float | None = None,
           model: lgb.Booster | None = None,
           min_precision: float = MIN_PRECISION) -> pl.DataFrame:
    t0 = time.time()
    scores = score(dataset, model)

    if threshold is None:
        if dataset == "test":
            raise ValueError("provide --threshold for test; tune on sample_5pct first")
        threshold = tune_threshold(dataset, scores, min_precision)

    pairs = decide_pairs(scores, threshold)

    # persist decision
    if dataset != "test":
        gt_s1, gt_pairs = C.load_ground_truth(dataset)
        folds = C.load_s1_folds(dataset)
        s1_valid = _eval_s1(dataset, folds)
        pairs_v   = pairs.join(s1_valid, on="s1_id", how="semi")
        gt_s1_v   = gt_s1.join(s1_valid, on="s1_id", how="semi")
        gt_pairs_v = gt_pairs.join(s1_valid, on="s1_id", how="semi")
        valid_f05 = evaluate.macro_f05(pairs_v, gt_s1_v, gt_pairs_v)

        rep = evaluate.score_report(pairs_v, gt_s1_v, gt_pairs_v)
        evaluate.print_report(rep, f"{dataset} valid fold (thresh={threshold:.2f})")

        decision = {
            "threshold": threshold,
            "rule": "prob>=t then one_owner",
            "tuned_on": dataset,
            "min_precision": min_precision,
            "valid_macro_f05": valid_f05,
            "model": str(MODEL_PATH := config.MODELS_DIR / "matcher.txt"),
        }
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        DECISION_PATH.write_text(json.dumps(decision, indent=2))
        print(f"decision saved → {DECISION_PATH}")

    print(f"done in {time.time()-t0:.1f}s  ({pairs.height:,} matches)")
    return pairs


def load_decision() -> dict:
    return json.loads(DECISION_PATH.read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sample_5pct", choices=list(C.DATASETS))
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min-precision", type=float, default=MIN_PRECISION)
    args = ap.parse_args()
    decide(args.dataset, threshold=args.threshold, min_precision=args.min_precision)
