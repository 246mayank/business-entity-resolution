"""Learned candidate pruning: the last candidate-generation stage (Track A).

Blocking (blocking.py) retrieves ~50 candidates per S1 with cheap sparse TF-IDF searches.
A small LightGBM pruner then scores every candidate with 17 cheap signals (blocking
similarities/ranks/flags, candidate counts, and a few fast string similarities) and keeps
at most K per S1 with pruner probability >= PMIN (~5 per S1). Only the kept pairs go to the
matching model, and they are what candidate_pairs.tsv contains.

    python -m src.prune --dataset train --fit      # fit on the train fold, then prune train
    python -m src.prune --dataset test             # prune test with the saved pruner

Pruning rewrites C.candidates_path(dataset) and C.features_path(dataset) in place with the
kept rows; the unpruned ("wide") versions are kept next to them as *.wide.parquet, and every
run reads from the wide files, so pruning is repeatable with other K / PMIN.
"""
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config
from . import contracts as C
from .sample import id_num

PRODUCER = "prune v1"

PRUNE_FEATURES = [
    "blk_name_sim", "blk_addr_sim", "blk_rank", "blk_rev_rank",
    "flag_name", "flag_addr", "flag_char4", "flag_reverse", "flag_combo",
    "n_cands", "n_s1_for_id", "rank_gap",
    "name_tsr", "addr_tsr", "name_ns_ratio", "place_jaccard", "hn_match",
]
K = 10            # at most this many candidates per S1
PMIN = 0.01       # and only those with pruner probability >= PMIN
BATCH = 5_000_000
PARAMS = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "verbose": -1,
          "seed": config.SEED, "n_jobs": -1}
N_ROUNDS = 200

MODEL_PATH = config.MODELS_DIR / "pruner.txt"
META_PATH = config.MODELS_DIR / "pruner_meta.json"


def wide_path(path: Path) -> Path:
    return path.with_name(path.stem + ".wide.parquet")


def _sources(dataset: str) -> tuple[Path, Path]:
    """(wide features, wide candidates); moves the current files aside on the first run."""
    out = []
    for path in (C.features_path(dataset), C.candidates_path(dataset)):
        wide = wide_path(path)
        if not wide.exists():
            os.replace(path, wide)
        out.append(wide)
    return out[0], out[1]


def reset(dataset: str) -> None:
    """Undo pruning: restore C2 to the retrieval output (wide, without key-route pairs) and drop the
    feature tables, so features can be rebuilt for a larger S1 share and pruned with the SAME pruner
    (the submitted model: pruner fitted on 20% of train S1, matcher trained on 40%)."""
    cand, wide = C.candidates_path(dataset), wide_path(C.candidates_path(dataset))
    if wide.exists():
        tmp = cand.with_name(cand.stem + ".tmp.parquet")
        pl.scan_parquet(wide).filter((pl.col("blk_flags") & C.FLAG_KEY) == 0).sink_parquet(tmp, compression="zstd")
        os.replace(tmp, cand)
        wide.unlink()
    for path in (C.features_path(dataset), wide_path(C.features_path(dataset))):
        path.unlink(missing_ok=True)
    print(f"prune reset {dataset}: {cand} restored to the retrieval candidates; feature tables removed")


def fit(dataset: str = "train") -> lgb.Booster:
    """Train the pruner on the train fold (minus the matcher's early-stopping S1s)."""
    t0 = time.time()
    feat_path = wide_path(C.features_path(dataset))
    if not feat_path.exists():
        feat_path = C.features_path(dataset)
    folds = C.load_s1_folds(dataset)
    fit_ids = folds.filter((pl.col("fold") == "train") & ((id_num("s1_id") // 100) % 10 != 1)).select("s1_id")
    _, gt = C.load_ground_truth(dataset)
    f = (pl.scan_parquet(feat_path).select(["s1_id", "id"] + PRUNE_FEATURES)
         .join(fit_ids.lazy(), on="s1_id", how="semi")
         .join(gt.lazy().with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1_id", "id"], how="left")
         .with_columns(pl.col("y").fill_null(0)).collect())
    model = lgb.train(PARAMS, lgb.Dataset(f.select(PRUNE_FEATURES).to_numpy().astype(np.float32),
                                          label=f["y"].to_numpy()), num_boost_round=N_ROUNDS)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    META_PATH.write_text(json.dumps({"trained_on": dataset, "features": PRUNE_FEATURES, "K": K, "PMIN": PMIN,
                                     "fit_pairs": f.height, "positives": int(f["y"].sum()),
                                     "written": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    print(f"pruner fit on {f.height:,} {dataset} pairs ({int(f['y'].sum()):,} positives) in {time.time() - t0:.0f}s")
    return model


def select(scored: pl.DataFrame, k: int = K, pmin: float = PMIN) -> pl.DataFrame:
    """Keep at most k pairs per S1 with p >= pmin (highest p first). scored: s1_id, id, p."""
    return (scored.filter(pl.col("p") >= pmin)
            .with_columns(pl.col("p").rank("ordinal", descending=True).over("s1_id").alias("_k"))
            .filter(pl.col("_k") <= k).drop("_k"))


def apply(dataset: str, model: lgb.Booster | None = None, k: int = K, pmin: float = PMIN) -> pl.DataFrame:
    """Prune C2 and the feature table of `dataset` to the kept pairs. Returns the kept pairs."""
    t0 = time.time()
    if model is None:
        model = lgb.Booster(model_file=str(MODEL_PATH))
    feat_src, cand_src = _sources(dataset)
    scan = pl.scan_parquet(feat_src)
    n = scan.select(pl.len()).collect().item()
    kept = []
    for off in range(0, n, BATCH):
        b = scan.slice(off, BATCH).select(["s1_id", "id"] + PRUNE_FEATURES).collect()
        p = model.predict(b.select(PRUNE_FEATURES).to_numpy().astype(np.float32)).astype(np.float32)
        kept.append(b.select("s1_id", "id").with_columns(pl.Series("p", p)).filter(pl.col("p") >= pmin))
    keep = select(pl.concat(kept), k, pmin)
    s1_before = scan.select(pl.col("s1_id").n_unique()).collect().item()

    scan.join(keep.lazy().select("s1_id", "id"), on=["s1_id", "id"], how="semi") \
        .sink_parquet(C.features_path(dataset), compression="zstd")
    cands = (pl.scan_parquet(cand_src).join(keep.lazy().select("s1_id", "id"), on=["s1_id", "id"], how="semi")
             .collect())
    C.write_candidates(cands, dataset, producer=f"{PRODUCER} (K={k}, PMIN={pmin})")
    print(f"prune {dataset}: {n:,} -> {keep.height:,} pairs; per S1 {n / s1_before:.1f} -> "
          f"{keep.height / s1_before:.2f} ({s1_before - keep['s1_id'].n_unique():,} S1 left without candidates) "
          f"in {time.time() - t0:.0f}s")
    return keep


def report(dataset: str) -> dict:
    """Recall of the pruned vs wide candidates on the valid fold (S1 that got features only)."""
    from .evaluate import blocking_report
    gt_s1, gt = C.load_ground_truth(dataset)
    folds = C.load_s1_folds(dataset)
    wide = pl.scan_parquet(wide_path(C.features_path(dataset))).select("s1_id", "id")
    featurized = wide.select("s1_id").unique().collect()
    valid = gt_s1.join(folds.filter(pl.col("fold") == "valid").select("s1_id"), on="s1_id", how="semi") \
                 .join(featurized, on="s1_id", how="semi")
    out = {}
    for name, pairs in (("wide", wide.collect()), ("pruned", pl.read_parquet(C.features_path(dataset), columns=["s1_id", "id"]))):
        out[name] = blocking_report(pairs, valid, gt)
    return out


if __name__ == "__main__":
    import argparse

    from .evaluate import print_report

    ap = argparse.ArgumentParser(description="Learned candidate pruning (last candidate-generation stage).")
    ap.add_argument("--dataset", default="train", choices=C.DATASETS)
    ap.add_argument("--fit", action="store_true", help="fit the pruner on this dataset first (not for test)")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--pmin", type=float, default=PMIN)
    ap.add_argument("--reset", action="store_true", help="undo pruning (restore the wide candidates) and exit")
    args = ap.parse_args()
    if args.reset:
        reset(args.dataset)
        raise SystemExit(0)
    if args.fit and args.dataset == "test":
        ap.error("the pruner cannot be fitted on test (no labels)")
    m = fit(args.dataset) if args.fit else None
    apply(args.dataset, m, args.k, args.pmin)
    if args.dataset != "test":
        for name, rep in report(args.dataset).items():
            print_report(rep, f"{args.dataset} valid fold, {name} candidates")
