# Business Entity Resolution — pipeline

All commands run from this folder (`code/business_entity_resolution/`). Data is
expected at `../../dataset/` (the organisers' `student_resource/dataset/`); set
`BER_ROOT` to the folder containing `dataset/` to use another location.

## Setup
```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
```

## Steps
| Step | Command | Output |
|---|---|---|
| 1. Cache raw TSVs as Parquet | `python -m src.io_utils` | `cache/*.parquet` (~1 GB, ~25 s) |
| 2. Build 5% dev sample | `python -m src.sample` | `cache/sample_5pct/` |
| 2b. Stub C1/C2 for the sample (until real normalize/blocking land) | `python -m src.stub` | `cache/norm/sample_5pct_*`, `cache/candidates/sample_5pct.parquet` (~1 min) |
| 3. Score a submission on train | `python -m src.evaluate --matching FILE [--candidates FILE]` | macro F0.5, blocking recall |
| Tests | `python -m pytest -q tests` | |

Blocking, features, training and prediction (`blocking.py`, `features.py`,
`train.py`, `predict.py`, `run_pipeline.py`) are still to be implemented.

## Modules
- `contracts.py` — stage interfaces C1–C4 (columns, dtypes, paths), validators, `write_*`/`load_*` helpers, dataset loaders (`load_records`, `load_s1_folds`, `load_ground_truth`). See `docs/plan/00_combined_plan.md` §3.
- `stub.py` — throwaway contract-valid C1/C2 for `sample_5pct`.
- `config.py` — paths, seeds, sample/validation bucketing (by numeric entity id, deterministic).
- `io_utils.py` — TSV reading (all strings, no quoting/NA inference), Parquet cache, submission writers.
- `sample.py` — dev sample: 5% of train S1 + all their matches + 5% of other S2/S3 records; `fold` column marks the 20% validation split.
- `evaluate.py` — competition metric (per-entity F0.5, macro-averaged, singletons included) and blocking recall / oracle ceiling.
