"""End-to-end pipeline: raw data -> output/matching_results.tsv + output/candidate_pairs.tsv.

    python -m src.run_pipeline --dataset sample_5pct              # dev loop: all steps, report on the valid fold
    python -m src.run_pipeline --dataset train --steps features train predict write --s1-pct 20
    python -m src.run_pipeline --dataset test --steps features predict write validate --model-from train

Steps (in order; --steps picks a subset):
  normalize  C1 via src.normalize          (subprocess: frees memory afterwards)
  block      C2 via src.blocking           (subprocess)
  features   pair features via src.features (--s1-pct: only that % of S1, for full train)
  prune      learned candidate pruning via src.prune: fit on train/sample, keep <= K per S1;
             rewrites C2 + features to the kept pairs (wide copies kept as *.wide.parquet)
  train      src.train.fit, then src.predict.decide tunes the threshold on the valid fold
             and writes models/decision.json (not for test)
  predict    score with models/matcher.txt, threshold from models/decision.json (or
             --threshold), one-owner -> final pairs
  write      both TSVs over all S1 ids of the dataset; report F0.5 on the valid fold
  validate   organisers' validator (test only)

Test outputs go to output/ (the submission); other datasets to output/dev/<dataset>/,
so a dev run never overwrites the submission files. When features were built for a
subset of S1 (--s1-pct), dev outputs and the F0.5 report cover that subset only.
"""
import json
import subprocess
import sys
import time

import polars as pl

from . import config
from . import contracts as C
from . import io_utils as io
from .evaluate import print_report, score_report

STEPS = ("normalize", "block", "features", "prune", "train", "predict", "write", "validate")
PIPE_DIR = config.CACHE_DIR / "pipeline"


def final_pairs_path(dataset: str):
    return PIPE_DIR / f"final_{dataset}.parquet"


def output_dir(dataset: str):
    return config.OUTPUT_DIR if dataset == "test" else config.OUTPUT_DIR / "dev" / dataset


def _run_module(module: str, *args: str) -> None:
    subprocess.run([sys.executable, "-m", module, *args], check=True, cwd=config.ROOT / "code" / "business_entity_resolution")


# ---------------------------------------------------------------- steps

def step_normalize(dataset: str) -> None:
    splits = [dataset] if dataset != "test" else ["train", "test"]  # test features fit on train C1
    for split in splits:
        _run_module("src.normalize", "--splits", split)


def step_block(dataset: str) -> None:
    _run_module("src.blocking", "--dataset", dataset)


def step_features(dataset: str, s1_pct: int | None = None) -> None:
    if dataset == "test" and s1_pct is not None:
        raise SystemExit("--s1-pct is not allowed for test: every test S1 needs predictions")
    from . import features
    features.build_features(dataset, s1_pct)


def step_prune(dataset: str, model_from: str) -> None:
    from . import prune
    if dataset == "test":
        meta = json.loads(prune.META_PATH.read_text()) if prune.META_PATH.exists() else {}
        if meta.get("trained_on") != model_from:
            raise SystemExit(f"models/pruner.txt was fitted on {meta.get('trained_on')!r}, not {model_from!r}: "
                             "run the prune step on --model-from first")
        prune.apply(dataset)
    else:
        prune.apply(dataset, prune.fit(dataset))


def step_train(dataset: str) -> None:
    if dataset == "test":
        print("train: skipped for test (use --model-from)")
        return
    from . import predict, train
    model = train.fit(dataset)
    predict.decide(dataset, model=model)           # tunes on the valid fold, writes models/decision.json


def _saved_threshold(model_from: str) -> float:
    from . import predict
    try:
        decision = predict.load_decision()
    except FileNotFoundError:
        raise SystemExit("no models/decision.json: run the train step or pass --threshold")
    if decision.get("tuned_on") != model_from:
        raise SystemExit(f"models/decision.json was tuned on {decision.get('tuned_on')!r}, not {model_from!r}: "
                         "retrain on --model-from or pass --threshold")
    return float(decision["threshold"])


def step_predict(dataset: str, model_from: str, threshold: float | None) -> None:
    from . import predict
    if threshold is None:
        threshold = _saved_threshold(model_from)
    scores = predict.score(dataset)                # always rescore: the model may have changed
    pairs = predict.decide_pairs(scores, threshold)  # threshold + one-owner, C4-checked against the scored pairs
    PIPE_DIR.mkdir(parents=True, exist_ok=True)
    pairs.write_parquet(final_pairs_path(dataset))
    print(f"predict: {pairs.height:,} matches at threshold {threshold:.3f} (model from {model_from})")


def _featurized_s1(dataset: str) -> pl.DataFrame:
    """S1 ids of the dataset that got features (all of them unless --s1-pct was used)."""
    from . import features
    s1 = C.load_records(dataset, 1, ["entity_id"]).rename({"entity_id": "s1_id"})
    return s1.filter(features.in_subsample(features.featurized_s1_pct(dataset)))


def _load_cands(dataset: str, s1: pl.DataFrame) -> pl.DataFrame:
    """C2 pairs of the given S1 ids only (lazy scan: full train C2 is 108M pairs)."""
    return (pl.scan_parquet(C.candidates_path(dataset)).select("s1_id", "id")
            .join(s1.lazy(), on="s1_id", how="semi").collect())


def step_write(dataset: str) -> None:
    s1 = _featurized_s1(dataset)
    s1_ids = s1["s1_id"]
    cands = _load_cands(dataset, s1)
    pairs = pl.read_parquet(final_pairs_path(dataset))
    C.check_c4(pairs, cands)
    out = output_dir(dataset)
    io.write_id_lists(pairs, s1_ids, out / "matching_results.tsv", io.MATCH_COL)
    io.write_id_lists(cands, s1_ids, out / "candidate_pairs.tsv", io.CAND_COL)
    print(f"write: {out / 'matching_results.tsv'} ({pairs.height:,} matches), "
          f"{out / 'candidate_pairs.tsv'} ({cands.height:,} candidates), {s1_ids.len():,} S1 rows each")
    if dataset != "test":
        gt_s1, gt_pairs = C.load_ground_truth(dataset)
        folds = C.load_s1_folds(dataset)
        valid = folds.filter(pl.col("fold") == "valid").join(s1, on="s1_id", how="semi")
        rep = score_report(pairs, gt_s1.join(valid.select("s1_id"), on="s1_id", how="semi"), gt_pairs,
                           valid.select(pl.col("s1_id").alias("entity_id"), "country"))
        print_report(rep, f"{dataset} valid fold")


def step_validate(dataset: str) -> None:
    if dataset != "test":
        print("validate: only for test")
        return
    out = output_dir(dataset)
    subprocess.run([sys.executable, "utils/validate_submission.py",
                    "--matching", str(out / "matching_results.tsv"),
                    "--candidate", str(out / "candidate_pairs.tsv"),
                    "--test-dir", "dataset/test"], check=True, cwd=config.ROOT)


def run(dataset: str, steps=STEPS, model_from: str | None = None, threshold: float | None = None,
        s1_pct: int | None = None) -> None:
    model_from = model_from or dataset
    if dataset == "test" and model_from == "test":
        raise SystemExit("--model-from is required for the test set (e.g. sample_5pct or train)")
    for step in STEPS:
        if step not in steps:
            continue
        t = time.time()
        print(f"== {step} ({dataset})", flush=True)
        if step == "predict":
            step_predict(dataset, model_from, threshold)
        elif step == "features":
            step_features(dataset, s1_pct)
        elif step == "prune":
            step_prune(dataset, model_from)
        else:
            globals()[f"step_{step}"](dataset)
        print(f"   {step} done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run the entity-resolution pipeline end to end.")
    ap.add_argument("--dataset", default="sample_5pct", choices=C.DATASETS)
    ap.add_argument("--steps", nargs="+", default=list(STEPS), choices=STEPS)
    ap.add_argument("--model-from", default=None, choices=[d for d in C.DATASETS if d != "test"],
                    help="dataset the model and threshold were trained on (required for test)")
    ap.add_argument("--threshold", type=float, default=None, help="override the saved decision threshold")
    ap.add_argument("--s1-pct", type=int, default=None, help="features for only this %% of S1 (full train)")
    args = ap.parse_args()
    run(args.dataset, args.steps, args.model_from, args.threshold, args.s1_pct)
