# Track B — Yash: Features, Matching Model, Decision, Packaging

Branch: **`yash`**. Read first: [00_combined_plan.md](00_combined_plan.md) (contracts C1–C4, file ownership, workflow) and [CLAUDE.md](../../CLAUDE.md).

**You own:** `src/features.py`, `src/train.py`, `src/predict.py`, `tests/test_features.py`, `tests/test_train.py`, `tests/test_predict.py`, `notebooks/B_*.ipynb`, `requirements/matching.txt`, `docs/method/matching.md`, `docs/experiments/B_log.md`, `make_submission.ps1` / `make_submission.sh`, `docs/REPRODUCE.md`.
**Never edit:** `normalize.py`, `blocking.py`, `run_pipeline.py`, `contracts.py`, `stub.py`, shared files (`config.py`, `io_utils.py`, `evaluate.py`, `sample.py`, root `requirements.txt`, `CLAUDE.md`). Need a change there → ask Mayank.

**Your north-star metric:** macro F0.5 on the `sample_5pct` valid fold (then full-train valid fold), reported overall / US / India / singletons, relative to the oracle ceiling Mayank’s candidates allow (`evaluate.blocking_report(...)["oracle_macro_f05"]`). Your job: close the gap between model F0.5 and the oracle, and make it hold for an unseen country (France).

---

## B0 — Setup (day 1)
1. Accept the GitHub invite. Clone, then create your branch:
   ```bash
   git clone https://github.com/246mayank/business-entity-resolution.git
   cd business-entity-resolution
   git checkout -b yash origin/main && git push -u origin yash
   ```
2. Get the dataset zip from the portal (or the link Mayank shares), copy its `dataset/` folder into the repo root.
3. Python 3.13 venv in the repo root, `pip install -r code/business_entity_resolution/requirements.txt`.
4. From `code/business_entity_resolution/`: `python -m src.io_utils`, `python -m src.sample`, `python -m pytest -q tests`. (Set `PYTHONIOENCODING=utf-8` on Windows.)
5. Review Mayank’s Phase-0 PR (contracts + stub). After it merges: `git merge origin/main`, run `python -m src.stub` → you now have C1/C2 for `sample_5pct`.
6. Read `src/evaluate.py` — use `score_report` for everything; never write your own metric.

## B1 — `src/features.py` v1
Public API: `build_features(dataset: str) -> pl.DataFrame` (also writes `cache/features/{dataset}.parquet`); `FEATURES: list[str]` (the model’s column list); CLI `python -m src.features --dataset sample_5pct`.
Input: C2 joined to C1 of S1 (left side) and C1 of S2/S3 (right side). Output: C2 keys + feature columns (float32), **no label** (labels are joined in `train.py` from GT).

Feature groups (all country-agnostic — **never** use `country` as a feature, it breaks on France):
- **Name** (on `name_core`, and a few on `name_clean`): rapidfuzz `ratio`, `partial_ratio`, `token_sort_ratio`, `token_set_ratio`, `JaroWinkler.normalized_similarity`, `Levenshtein.normalized_distance`; token Jaccard; overlap of rarest token (IDF-weighted Jaccard — IDF from the country’s S1+S2+S3 token counts, computed within the dataset, not hard-coded); first-token equal; acronym match (initials of one == other’s token, e.g. `ibm` vs `international business machines`); numeric tokens in name equal / conflict; length ratio.
- **Legal form**: equal / both non-empty and different / one missing.
- **Address** (on `addr_clean`): `token_set_ratio`, `partial_ratio`, token Jaccard, IDF-weighted Jaccard; `postcode` equal / conflict / missing-one / missing-both; `house_no` equal / conflict / missing; either address empty flag.
- **Blocking signals from C2**: `blk_name_sim`, `blk_addr_sim`, `blk_rank`, `blk_rev_rank`, each bit of `blk_flags`.
- **Context (group) features** — big precision wins, computed with Polars window functions:
  - per S1: number of candidates; this pair’s name-sim rank and gap to the best; how many candidates have `name_sim ≥ 0.9`.
  - per record `id`: number of S1s it is a candidate for; is this S1 its best by name/address sim; gap to its 2nd best S1 (records have exactly one owner — a record that fits two S1s equally is risky).
- **Source flag**: `id` from S2 vs S3 (the sources have different noise styles — legit, generalises to France).
- **Script flag**: either side `is_native_script`.
Implementation notes:
- Compute rapidfuzz features on **unique string pairs** (dedupe first), in chunks, with `multiprocessing.Pool` (Windows: guard with `if __name__ == "__main__"`), then join back. Target: ≥ 1M pairs/min on 8 cores. Full train may be 50–100M pairs → you will train on a subsample (B2), but test scoring needs all test pairs.
- float32 everywhere; write Parquet per chunk if memory grows.
Tests: tiny hand-made C1/C2 frames → known feature values (identical strings → 1.0, disjoint → 0), no NaNs, row count preserved, order of pairs doesn’t matter.

## B2 — `src/train.py`
Public API: `fit(dataset: str, fold: str = "train") -> Path` → writes `models/matcher.txt` (LightGBM text model) and `models/matcher_meta.json` (feature list, params, train dataset + commit hash, valid metrics); CLI `python -m src.train --dataset sample_5pct`.
- Labels: pair ∈ GT pairs → 1 else 0 (join `io_utils.load_gt()` / sample GT).
- Split: **only S1 entities with `fold == "train"`** (see `sample.with_fold`); never let a valid-fold S1 into training. For internal early stopping use a GroupKFold / group split by `s1_id` inside the train fold.
- Model: LightGBM binary (MIT). Start: `num_leaves=63, learning_rate=0.05, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, n_estimators≤2000 with early stopping, seed=config.SEED`. Class imbalance: don’t reweight at first — we tune the threshold anyway.
- Full train (M3): C2 for full train may be ~100M rows. Subsample **by S1 entity** (e.g. 20–30% of train-fold S1s, keeping all their candidates) — never subsample rows within an entity, context features would lie.
- Log feature importance (gain) in `B_log.md` every version.

## B3 — `src/predict.py`
Public API:
- `score(dataset: str) -> pl.DataFrame` → C3 (`s1_id, id, prob`), writes `cache/scores/{dataset}.parquet`.
- `decide(scores: pl.DataFrame, params: dict | None = None) -> pl.DataFrame` → C4 pairs; must pass `contracts.check_c4`.
- `tune(dataset: str) -> dict` → finds decision params maximising macro F0.5 on the valid fold; saves to `models/decision.json`.
Decision logic v1 (order matters):
1. **One-owner**: for each `id`, keep only the S1 with the highest `prob` (ties → drop both unless one clearly wins). Optionally require a margin over the 2nd best S1.
2. **Threshold**: keep pairs with `prob ≥ t`. Tune `t` on a grid (0.3…0.95) against `evaluate.macro_f05` on the valid fold — **not** AUC/logloss.
3. Relative cut per S1: drop pairs with `prob < r × max_prob_of_this_S1` (tune `r`).
v2 (M4): **expected-F0.5 decision** — per S1, sort candidates by prob, choose the top-k that maximises expected F0.5 given the probabilities (including k = 0 → the singleton case; expected score of predicting nothing = P(no true matches) ≈ Π(1−p)). This directly optimises the metric and handles singletons naturally. Needs reasonably calibrated probs — check a reliability curve; if off, fit isotonic calibration on the valid fold of the train split (not on test).
Tests: one-owner never assigns an `id` twice; output ⊆ input pairs; `decide` on empty input returns empty; threshold 1.01 → all empty.

## B4 — Generalisation to France (M1 onwards, cheap and important)
- **Leave-one-country-out**: train on US-only train fold, evaluate on India valid fold, and vice versa. Report in `B_log.md` next to the in-country score. Features whose importance collapses cross-country or that hurt LOCO → rework or drop.
- Keep decision params that are robust across both LOCO directions; prefer a single global threshold over per-country ones (France has no labels to tune on).
- Before the first upload, eyeball ~100 test France predictions (top probs, borderline probs) in `notebooks/B_02_france_check.ipynb`.

## B5 — Error analysis loop (M4) `notebooks/B_01_error_analysis.ipynb`
On the valid fold, bucket errors:
- False positives: same generic name, different address; chains/franchises (same name, many branches); one-owner conflicts.
- False negatives: in C2 but scored low (your problem) vs not in C2 (Mayank’s — send him examples with ids, don’t fix it in features).
Each bucket → a feature or decision change → measured in `B_log.md`. Merge only what improves valid F0.5.

## B6 — Optional semantic similarity (M4, only if time/compute allows)
Small multilingual sentence-embedding model, MIT/Apache and ≪ 8B params, e.g. `intfloat/multilingual-e5-small` (MIT) or `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0). Embed unique `name_core` strings once (≈ 10M unique over train+test — GPU on AWS recommended; CPU is hours), cosine similarity as one extra feature. Record model name + license in `docs/method/matching.md` and pin in `requirements/matching.txt`. Only keep if it improves the full-train valid fold and LOCO.

## B7 — Packaging and reproducibility (M5)
- `make_submission.ps1` (+ `.sh`): builds `<team_name>_submission.zip` with `output/` (both TSVs), `code/business_entity_resolution/` (src, README, requirements) and `Documentation_template.md` — excluding caches, data, venv, notebooks, `__pycache__`.
- `docs/REPRODUCE.md`: exact commands from a clean clone + dataset to both output files, with timings and hardware.
- **Reproduction test**: fresh clone in a new folder, new venv, run the documented commands, diff the produced `matching_results.tsv` against the uploaded one. Fix non-determinism (seeds, sort orders, LightGBM `deterministic=True`, `num_threads` fixed).
- Write `docs/method/matching.md` (features, model, decision, validation, LOCO results) — Mayank folds it into the final template.

## What you need from Mayank (and when)
| When | What | Until then use |
|---|---|---|
| day 1–2 | Phase-0: `contracts.py` + `stub.py` merged to `main` | read the contracts in the combined plan §3 |
| M1→M2 | real C1/C2 for `sample_5pct` | stub C1/C2 |
| M3 | full `train` + `test` C1/C2 (local or S3) | sample |
If blocked on anything else — message him; don’t edit his files.

## Suggested order
B0 → B1 v1 (on stub) → B2 → B3 v1 → first valid-fold F0.5 in `B_log.md` → B4 LOCO → merge `yash → main` → switch to real C2 (M2) → M3 full train → B5/B3-v2/B6 loop → B7.
