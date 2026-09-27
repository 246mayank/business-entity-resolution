"""Train a LightGBM matcher on pair features and save the model.

Run:
    python -m src.train [--dataset sample_5pct|train]
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config, contracts as C
from .features import FEATURE_COLS
from .sample import id_num

MODEL_PATH = config.MODELS_DIR / "matcher.txt"
META_PATH  = config.MODELS_DIR / "matcher_meta.json"

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
N_ROUNDS = 6000  # full train hit 3000 still improving; early stopping ends it sooner when done
EARLY_STOPPING = 80


def _pairs_for_ids(feat: pl.DataFrame, gt_pairs: pl.DataFrame,
                   s1_ids: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    f = feat.join(s1_ids, on="s1_id", how="semi")
    labels = (f.select("s1_id", "id")
               .join(gt_pairs.select("s1_id", "id").with_columns(pl.lit(1).alias("label")),
                     on=["s1_id", "id"], how="left")
               .with_columns(pl.col("label").fill_null(0)))
    X = f.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y = labels["label"].to_numpy().astype(np.float32)
    return X, y


def fit(dataset: str = "sample_5pct") -> lgb.Booster:
    t0 = time.time()
    feat = pl.read_parquet(C.features_path(dataset))
    gt_s1, gt_pairs = C.load_ground_truth(dataset)
    folds = C.load_s1_folds(dataset)

    tr_ids = folds.filter(pl.col("fold") == "train").select("s1_id")

    # split train fold: ~10% for early stopping, rest for fitting
    es_ids  = tr_ids.filter((id_num("s1_id") // 100) % 10 == 1)
    fit_ids = tr_ids.join(es_ids, on="s1_id", how="anti")

    X_fit, y_fit = _pairs_for_ids(feat, gt_pairs, fit_ids)
    X_es,  y_es  = _pairs_for_ids(feat, gt_pairs, es_ids)

    pos = y_fit.sum()
    neg = len(y_fit) - pos
    print(f"fit:  {len(X_fit):,} pairs, {int(pos):,} positives ({pos/len(y_fit)*100:.2f}%)")
    print(f"ES:   {len(X_es):,}  pairs, {int(y_es.sum()):,} positives")

    dtrain = lgb.Dataset(X_fit, label=y_fit, feature_name=FEATURE_COLS, free_raw_data=False)
    des    = lgb.Dataset(X_es,  label=y_es,  feature_name=FEATURE_COLS, free_raw_data=False,
                         reference=dtrain)

    params = dict(LGB_PARAMS, scale_pos_weight=neg / max(pos, 1))
    callbacks = [lgb.early_stopping(EARLY_STOPPING, verbose=False),
                 lgb.log_evaluation(50)]

    model = lgb.train(params, dtrain, num_boost_round=N_ROUNDS,
                      valid_sets=[des], callbacks=callbacks)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))

    meta = {
        "dataset": dataset,
        "n_rounds": model.num_trees(),
        "features": FEATURE_COLS,
        "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fit_pairs": len(X_fit),
        "es_pairs": len(X_es),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"model saved → {MODEL_PATH}  ({time.time()-t0:.1f}s)")
    return model


def load_model() -> lgb.Booster:
    return lgb.Booster(model_file=str(MODEL_PATH))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sample_5pct", choices=["sample_5pct", "train"])
    args = ap.parse_args()
    fit(args.dataset)
