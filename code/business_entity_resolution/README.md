# Business Entity Resolution — pipeline

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the organisers'
data, using only what is in this folder. Methodology: `Documentation_template.md` at the package root.

## Environment
- Python **3.13** (tested with 3.13.5 on Windows 11 and 3.13.15 on Ubuntu 24.04), 16 GB RAM is enough (every step is chunked/streamed).
- Linux only: LightGBM needs the OpenMP runtime — `sudo apt-get install -y libgomp1`.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # pinned; pulls requirements/blocking.txt and requirements/matching.txt
```

Data: the organisers' `dataset/` folder (train + test TSVs) must sit two levels up, i.e. `<root>/dataset/`,
with this folder at `<root>/code/business_entity_resolution/`. Set `BER_ROOT=<root>` to use another
location. All intermediate files go to `<root>/cache/` and `<root>/models/`; outputs to `<root>/output/`.
Run every command from this folder.

## Reproduce end to end

| # | Step | Command | Time* | Output |
|---|---|---|---|---|
| 1 | Cache raw TSVs as Parquet | `python -m src.io_utils` | 30 s | `cache/*.parquet` |
| 2 | Normalise (learns the Indic dictionary + place map from train first) | `python -m src.normalize` | 5–8 min | `cache/norm/*` |
| 3 | Candidate retrieval, train | `python -m src.blocking --dataset train` | 70 min | `cache/candidates/train.parquet` (+ recall report) |
| 4 | Candidate retrieval, test | `python -m src.blocking --dataset test` | 52 min | `cache/candidates/test.parquet` |
| 5 | Train: features (20% of S1) → pruner → matcher → threshold | `python -m src.run_pipeline --dataset train --steps features prune train predict write --s1-pct 20` | see note | `models/`, valid-fold report |
| 6 | Test: features → prune → predict → write → validate | `python -m src.run_pipeline --dataset test --steps features prune predict write validate --model-from train` | see note | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |

\* On a 16 GB, 8-core laptop. Steps 5–6 first build the cheap pruning features on the wide candidate set
(train 47 min for 20% of S1, test ~3 h), prune to ≤10 candidates per S1 (~5–6), then rebuild the full
features on the kept pairs only (minutes) — the matcher trains and predicts on exactly `candidate_pairs.tsv`.

Step 6 ends with the organisers' validator (`utils/validate_submission.py`, run from `<root>`), which must print `PASS`.

## Stages and modules

| Stage | Module | Notes |
|---|---|---|
| I/O | `src/io_utils.py` | TSV read (all strings, no quoting, empty stays ""), Parquet cache, submission writers |
| Contracts | `src/contracts.py` | C1 normalised records, C2 candidates, C3 scores, C4 final pairs: schemas + validators (ids exist, same country, one owner per S2/S3 id) |
| Normalise | `src/normalize.py` | accent folding, junk/DBA/domain/legal-form handling, learned Indic→English dictionary, learned per-country place map |
| Retrieve | `src/blocking.py` | per-country sparse TF-IDF top-K: name, address, combined, place-conditioned, char 4-grams, reverse |
| Features | `src/features.py` | 48 pair features, computed per S1 chunk and streamed to one Parquet |
| Prune | `src/prune.py` | learned pruner (17 cheap features) keeps ≤10 candidates/S1 with p ≥ 0.01 — the final candidate set; wide copies kept as `*.wide.parquet`; the pipeline then rebuilds the full features on the kept pairs |
| Match | `src/train.py`, `src/predict.py` | LightGBM matcher; threshold tuned on the valid fold for macro F0.5; one-owner assignment; `models/decision.json` |
| Evaluate | `src/evaluate.py` | competition metric (per-S1 F0.5, macro, singletons included), blocking recall and oracle ceiling |
| Orchestrate | `src/run_pipeline.py` | steps `normalize block features prune train predict write validate`; dev outputs go to `output/dev/<dataset>/`, test outputs to `output/` |
| Dev only | `src/sample.py`, `src/stub.py` | 5% dev sample with valid fold; throwaway stage stubs |

Validation fold: S1 entities with `(numeric id // 100) % 5 == 0` (20%); the matcher early-stops on a further
10% of the train fold, so the valid fold is only used for threshold tuning and reporting.

## Tests
```bash
python -m pytest -q tests
```
