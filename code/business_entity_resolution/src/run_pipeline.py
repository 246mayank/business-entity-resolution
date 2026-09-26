"""End-to-end pipeline: raw data -> output/matching_results.tsv + output/candidate_pairs.tsv.

    python -m src.run_pipeline --dataset sample_5pct              # dev loop: all steps, report on the valid fold
    python -m src.run_pipeline --dataset test --model-from sample_5pct
    python -m src.run_pipeline --dataset test --steps predict write validate --threshold 0.6

Steps (in order; --steps picks a subset):
  normalize  C1 via src.normalize          (subprocess: frees memory afterwards)
  block      C2 via src.blocking           (subprocess)
  features   pair features via src.features
  train      src.train.fit + threshold tuning on the valid fold (not for test)
  predict    score C2 with the saved model, threshold, one-owner -> final pairs
  write      both TSVs over all S1 ids of the dataset; report F0.5 on the valid fold
  validate   organisers' validator (test only)

Test outputs go to output/ (the submission); other datasets to output/dev/<dataset>/,
so a dev run never overwrites the submission files.
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

STEPS = ("normalize", "block", "features", "train", "predict", "write", "validate")
PIPE_DIR = config.CACHE_DIR / "pipeline"


def threshold_path(model_dataset: str):
    return PIPE_DIR / f"threshold_{model_dataset}.json"


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


def step_features(dataset: str) -> None:
    from . import features
    features.build_features(dataset)


def step_train(dataset: str) -> None:
    if dataset == "test":
        print("train: skipped for test (use --model-from)")
        return
    from . import predict, train
    model = train.fit(dataset)
    scores = predict.score(dataset, model)
    t = predict.tune_threshold(dataset, scores)
    PIPE_DIR.mkdir(parents=True, exist_ok=True)
    threshold_path(dataset).write_text(json.dumps(
        {"model_dataset": dataset, "threshold": t, "written": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))


def step_predict(dataset: str, model_from: str, threshold: float | None) -> None:
    from . import predict
    if threshold is None:
        path = threshold_path(model_from)
        if not path.exists():
            raise SystemExit(f"no saved threshold for model trained on {model_from!r}: run the train step or pass --threshold")
        threshold = json.loads(path.read_text())["threshold"]
    scores = predict.score(dataset)                      # always rescore: the model may have changed
    pairs = predict._one_owner(scores.filter(pl.col("prob") >= threshold))
    C.check_c4(pairs, C.load_candidates(dataset))
    PIPE_DIR.mkdir(parents=True, exist_ok=True)
    pairs.write_parquet(final_pairs_path(dataset))
    print(f"predict: {pairs.height:,} matches at threshold {threshold:.3f} (model from {model_from})")


def step_write(dataset: str) -> None:
    s1_ids = C.load_records(dataset, 1)["entity_id"]
    cands = C.load_candidates(dataset).select("s1_id", "id")
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
        valid = folds.filter(pl.col("fold") == "valid")
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


def run(dataset: str, steps=STEPS, model_from: str | None = None, threshold: float | None = None) -> None:
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
    args = ap.parse_args()
    run(args.dataset, args.steps, args.model_from, args.threshold)
