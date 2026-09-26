"""Train a LightGBM matcher on pair features and save the model.

Run:
    python -m src.train [--dataset sample_5pct|train]
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config, contracts as C
from .features import FEATURE_COLS

MODEL_PATH = config.MODELS_DIR / "matcher.txt"
META_PATH  = config.MODELS_DIR / "matcher_meta.json"

# LightGBM hyperparameters (tuned for macro F0.5 on sample)
LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "seed": config.SEED,
    "verbose": -1,
}
N_ROUNDS = 1000
EARLY_STOPPING = 80


def _load_split(dataset: str, fold: str) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Return (X, y, pairs_df) for a fold of a dataset."""
    feat = pl.read_parquet(C.features_path(dataset))
    gt_s1, gt_pairs = C.load_ground_truth(dataset)
    folds = C.load_s1_folds(dataset)

    s1_in_fold = folds.filter(pl.col("fold") == fold).select("s1_id")
    feat_fold = feat.join(s1_in_fold, on="s1_id", how="semi")
    labels = feat_fold.select("s1_id", "id").join(
        gt_pairs.select("s1_id", "id").with_columns(pl.lit(1).alias("label")),
        on=["s1_id", "id"], how="left"
    ).with_columns(pl.col("label").fill_null(0))

    X = feat_fold.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y = labels["label"].to_numpy().astype(np.float32)
    return X, y, feat_fold


def fit(dataset: str = "sample_5pct") -> lgb.Booster:
    t0 = time.time()
    X_tr, y_tr, _ = _load_split(dataset, "train")
    X_vl, y_vl, _ = _load_split(dataset, "valid")

    pos = y_tr.sum()
    neg = len(y_tr) - pos
    print(f"train: {len(X_tr):,} pairs, {int(pos):,} positives ({pos/len(y_tr)*100:.2f}%)")
    print(f"valid: {len(X_vl):,} pairs, {int(y_vl.sum()):,} positives")

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_COLS,
                         free_raw_data=False)
    dvalid = lgb.Dataset(X_vl, label=y_vl, feature_name=FEATURE_COLS,
                         free_raw_data=False, reference=dtrain)

    params = dict(LGB_PARAMS, scale_pos_weight=neg / max(pos, 1))
    callbacks = [lgb.early_stopping(EARLY_STOPPING, verbose=False),
                 lgb.log_evaluation(50)]

    model = lgb.train(params, dtrain, num_boost_round=N_ROUNDS,
                      valid_sets=[dvalid], callbacks=callbacks)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))

    meta = {
        "dataset": dataset,
        "n_rounds": model.num_trees(),
        "features": FEATURE_COLS,
        "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_pairs": len(X_tr),
        "valid_pairs": len(X_vl),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"model saved → {MODEL_PATH}  ({time.time()-t0:.1f}s)")
    return model


def load_model() -> lgb.Booster:
    return lgb.Booster(model_file=str(MODEL_PATH))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sample_5pct",
                    choices=["sample_5pct", "train"])
    args = ap.parse_args()
    fit(args.dataset)
