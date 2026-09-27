# Reproducing the submission from a clean clone

The step-by-step commands, timings and module map are in
[`code/business_entity_resolution/README.md`](../code/business_entity_resolution/README.md).
This page is the checklist for doing it on a fresh machine and confirming the result.

## 1. Clone and install

```bash
git clone https://github.com/246mayank/business-entity-resolution.git
cd business-entity-resolution/code/business_entity_resolution
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
sudo apt-get install -y libgomp1                        # Linux only: OpenMP runtime for LightGBM
python -m pytest -q tests                               # expect all tests to pass
```

Python 3.13 (3.12 also works). 16 GB RAM is enough; every step is chunked or streamed.

## 2. Place the data

Unzip the organisers' data so the tree is:

```
business-entity-resolution/
├── dataset/
│   ├── train/   train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
│   └── test/    test_source1.tsv, test_source2.tsv, test_source3.tsv
└── code/business_entity_resolution/
```

Anywhere else works with `BER_ROOT=<folder holding dataset/>`.

## 3. Start from nothing learned

`cache/` and `models/` must not exist, or must be empty. In particular, `models/tfidf_*.joblib`
is reused when present, so a stale vectoriser from a dev run would change `name_cos` / `addr_cos`.

```bash
rm -rf ../../cache ../../models ../../output/*.tsv
```

## 4. Run

From `code/business_entity_resolution/`, run steps 1–6 of the code README in order:
cache → normalise → blocking (train, test) → train pipeline → test pipeline.
Blocking dominates the runtime (~2 h); the pipeline steps take ~1 h on train (20% of S1) and ~3 h on test.

## 5. Check the result

| Check | Where | Expected |
|---|---|---|
| Unit tests | `pytest -q tests` | all pass |
| Train candidate recall / oracle (valid fold, pruned) | blocking + prune reports | 0.9436 / 0.9791 |
| Matcher macro F0.5, full-train valid fold | train pipeline report, `models/decision.json` → `valid_macro_f05` | final model value in `Documentation_template.md` §5 |
| Test candidates per S1 | prune report | ~6.2 |
| Submission validity | test pipeline `validate` step | `PASS` |

Numbers should match to the 3rd decimal: every split is a deterministic function of the entity id,
and all random seeds are fixed (`config.SEED = 42`). LightGBM multithreading can still move the last
digits slightly.

## 6. Package

From the repository root:

```bash
./make_submission.sh <team_name>            # Windows: .\make_submission.ps1 -Team <team_name>
```

It re-runs the organisers' validator and refuses to build the zip if it fails. It warns if
`Documentation_template.md` still has `TBD` placeholders.
