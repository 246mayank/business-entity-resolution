"""Score candidates, tune threshold for macro F0.5, write output TSVs.

Run:
    python -m src.predict [--dataset sample_5pct|train|test] [--threshold 0.5]
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config, contracts as C, io_utils as io, evaluate
from .features import FEATURE_COLS
from .train import load_model, _load_split


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


# ------------------------------------------------------------------ threshold tuning

def _f05_at_threshold(scores: pl.DataFrame, gt_s1: pl.DataFrame,
                      gt_pairs: pl.DataFrame, thresh: float) -> float:
    pairs = scores.filter(pl.col("prob") >= thresh).select("s1_id", "id")
    return evaluate.macro_f05(pairs, gt_s1, gt_pairs)


def tune_threshold(dataset: str, scores: pl.DataFrame | None = None,
                   folds: tuple[str, ...] = ("valid",)) -> float:
    """Grid-search threshold on the given folds; returns the best threshold."""
    if scores is None:
        scores = C.load_scores(dataset)
    gt_s1, gt_pairs = C.load_ground_truth(dataset)
    fold_df = C.load_s1_folds(dataset)

    s1_eval = fold_df.filter(pl.col("fold").is_in(list(folds))).select("s1_id")
    scores_eval = scores.join(s1_eval, on="s1_id", how="semi")
    gt_s1_eval  = gt_s1.join(s1_eval, on="s1_id", how="semi")
    gt_pairs_eval = gt_pairs.join(s1_eval, on="s1_id", how="semi")

    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.02, 0.98, 0.02):
        f = _f05_at_threshold(scores_eval, gt_s1_eval, gt_pairs_eval, float(t))
        if f > best_f:
            best_f, best_t = f, float(t)
    print(f"best threshold: {best_t:.2f}  →  macro F0.5 = {best_f:.4f} on {folds}")
    return best_t


# ------------------------------------------------------------------ one-owner assignment

def _one_owner(pairs: pl.DataFrame) -> pl.DataFrame:
    """Each S2/S3 id may belong to at most one S1. Keep the highest-prob assignment."""
    return (pairs.sort("prob", descending=True)
            .unique(subset=["id"], keep="first")
            .select("s1_id", "id"))


# ------------------------------------------------------------------ decide

def decide(dataset: str, threshold: float | None = None,
           model: lgb.Booster | None = None) -> pl.DataFrame:
    t0 = time.time()

    # always rescore (model may have changed)
    scores = score(dataset, model)

    # tune threshold on valid fold if not given
    if threshold is None:
        if dataset == "test":
            raise ValueError("provide --threshold for test; tune on sample_5pct first")
        threshold = tune_threshold(dataset, scores)

    pairs = scores.filter(pl.col("prob") >= threshold)
    pairs = _one_owner(pairs)

    s1_ids = C.load_records(dataset, 1)["entity_id"]

    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    match_path = out_dir / "matching_results.tsv"
    cand_path  = out_dir / "candidate_pairs.tsv"

    io.write_id_lists(pairs, s1_ids, match_path, io.MATCH_COL)

    cands = C.load_candidates(dataset).select(
        pl.col("s1_id"), pl.col("id")
    )
    io.write_id_lists(cands, s1_ids, cand_path, io.CAND_COL)

    print(f"wrote {match_path.name}: {pairs.height:,} matches for {s1_ids.len():,} S1 entities")
    print(f"wrote {cand_path.name}")

    # report on non-test datasets
    if dataset != "test":
        gt_s1, gt_pairs = C.load_ground_truth(dataset)
        folds = C.load_s1_folds(dataset)
        s1_valid = folds.filter(pl.col("fold") == "valid").select("s1_id")
        rep = evaluate.score_report(
            pairs.join(s1_valid, on="s1_id", how="semi"),
            gt_s1.join(s1_valid, on="s1_id", how="semi"),
            gt_pairs.join(s1_valid, on="s1_id", how="semi"),
        )
        evaluate.print_report(rep, f"{dataset} valid fold (thresh={threshold:.2f})")

    print(f"done in {time.time()-t0:.1f}s")
    return pairs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sample_5pct",
                    choices=list(C.DATASETS))
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()
    decide(args.dataset, threshold=args.threshold)
