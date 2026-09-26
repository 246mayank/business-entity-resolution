# Combined Work Plan — Business Entity Resolution

Owners: **Mayank** (Track A: text normalisation + blocking + orchestration) and **Yash** (Track B: matching model + decision + packaging).
Personal task lists: [01_mayank_normalization_blocking.md](01_mayank_normalization_blocking.md) · [02_yash_matching_model.md](02_yash_matching_model.md).
Project rules and verified data facts: [CLAUDE.md](../../CLAUDE.md) — read it first.

---

## 1. Goal and scoring recap
For every test S1 entity (1,732,544 of them) output the S2/S3 records that are the same business.
Score = macro F0.5 per S1 entity (singletons count: empty prediction on a singleton = 1.0).
Precision matters 2× recall → the pipeline must be conservative.

Facts that shape the design (verified on train):
- Each S2/S3 record belongs to **at most one** S1 → one-owner assignment is a free precision gain.
- Matches always share `country` → all work is done per country.
- Test S1 = India 47%, US 38%, **France 15% (unseen in train)** → nothing may depend on country identity.
- Exact-name baseline: F0.5 0.364 (precision 0.46) → names are ambiguous; address evidence is essential.

## 2. Architecture

```
dataset/*.tsv
   │  io_utils.convert_all()                                    [shared, done]
   ▼
cache/{split}_source{n}.parquet
   │  normalize.normalize_all()                                 [Track A]
   ▼
cache/norm/{split}_source{n}.parquet         ── contract C1 ──►  used by A (blocking) and B (features)
   │  blocking.build_candidates(dataset)                        [Track A]
   ▼
cache/candidates/{dataset}.parquet           ── contract C2 ──►  used by B
   │  features.build_features(dataset)                          [Track B]
   ▼
cache/features/{dataset}.parquet
   │  train.fit()  → models/matcher.txt + models/matcher_meta.json   [Track B]
   │  predict.score() → cache/scores/{dataset}.parquet               [Track B]
   │  predict.decide() → pairs (s1_id, id)                           [Track B]
   ▼
output/matching_results.tsv + output/candidate_pairs.tsv   (io_utils.write_id_lists, run_pipeline)  [Track A]
```

`{dataset}` is one of `sample_5pct` (dev), `train` (full train), `test`.

## 3. Interface contracts (frozen after Phase 0 — change only via a "contract PR" both approve)

Contracts live in `src/contracts.py` (column names, dtypes, paths, `check_*()` validators). Producers must call the validator before writing; consumers may assume it passed. **Extra columns are allowed; contract columns may never be removed or renamed.**

### C1 — normalised records `cache/norm/{split}_source{n}.parquet` (producer: A)
| column | dtype | meaning |
|---|---|---|
| `entity_id` | str | unchanged |
| `country` | str | unchanged |
| `name_clean` | str | lowercase, ASCII (transliterated/accent-folded), punctuation→space, abbreviations expanded, single spaces |
| `name_core` | str | `name_clean` without legal-form and junk tokens (may be "" only if name was only a legal form) |
| `legal_form` | str | canonical legal form (`inc`, `llc`, `corp`, `ltd`, `pvt_ltd`, `llp`, `sarl`, `sas`, … or "") |
| `addr_clean` | str | same cleaning as names, abbreviations expanded, "" if missing |
| `postcode` | str | digits only (US 5, India 6, France 5) or "" |
| `house_no` | str | leading street/house number token or "" |
| `is_native_script` | bool | original name or address was not Latin script |

### C2 — candidates `cache/candidates/{dataset}.parquet` (producer: A)
| column | dtype | meaning |
|---|---|---|
| `s1_id` | str | S1 entity |
| `id` | str | S2-/S3- record (same country as `s1_id`) |
| `blk_name_sim` | float32 | best name similarity seen by blocking (0 if none) |
| `blk_addr_sim` | float32 | best address similarity seen by blocking (0 if none) |
| `blk_rank` | uint16 | best rank of `id` among this S1's retrieved records (1 = best) |
| `blk_rev_rank` | uint16 | best rank of `s1_id` among this record's retrieved S1s (1 = best; 0 if not computed) |
| `blk_flags` | uint8 | bitmask of blockers that produced the pair (1=name tfidf, 2=addr tfidf, 4=postcode key, 8=reverse, …) |

S1s with no candidates simply have no rows; `run_pipeline` adds their empty rows when writing the TSVs.

### C3 — scores `cache/scores/{dataset}.parquet` (producer: B)
`s1_id` str, `id` str, `prob` float32 — one row per C2 row.

### C4 — final pairs (producer: B, `predict.decide()` return value)
DataFrame `s1_id`, `id` — subset of C2. Each `id` appears at most once.

### Stubs so both tracks start on day 1
`src/stub.py` (Phase 0) produces *cheap but contract-valid* C1 and C2 for `sample_5pct`:
- C1 stub: lowercase + unidecode + punctuation strip; `name_core = name_clean`; empty `legal_form`, regex postcode/house_no.
- C2 stub: per country, pairs sharing the first 4 chars of `name_core` or the same postcode, capped at 50 per S1, plus similarity columns from rapidfuzz.

Track B develops against the stub; when Track A publishes real C1/C2 for the sample, B just re-runs — no code change.

## 4. File ownership (the mutex)

Only the owner edits a file. Need a change in someone else's file → open a GitHub issue or message; the owner does it.

Paths: `src/`, `tests/`, `requirements/` live under `code/business_entity_resolution/` (that folder ships in the submission zip); `docs/`, `notebooks/`, `.github/`, `CLAUDE.md`, `Documentation_template.md`, `make_submission.*` are at the repo root.

| Path | Owner |
|---|---|
| `src/normalize.py`, `src/blocking.py`, `src/run_pipeline.py` | **Mayank** |
| `src/features.py`, `src/train.py`, `src/predict.py` | **Yash** |
| `src/contracts.py`, `src/stub.py` | Mayank writes in Phase 0; afterwards contract PRs only |
| `src/config.py`, `src/io_utils.py`, `src/evaluate.py`, `src/sample.py` | shared-frozen (shared-change PRs only) |
| `tests/test_normalize.py`, `tests/test_blocking.py`, `tests/test_pipeline.py` | Mayank |
| `tests/test_features.py`, `tests/test_train.py`, `tests/test_predict.py` | Yash |
| `notebooks/A_*.ipynb` | Mayank |
| `notebooks/B_*.ipynb` | Yash |
| `requirements/blocking.txt` | Mayank |
| `requirements/matching.txt` | Yash |
| `requirements.txt` (only `-r` includes + shared pins) | shared-frozen |
| `docs/method/blocking.md`, `docs/experiments/A_log.md` | Mayank |
| `docs/method/matching.md`, `docs/experiments/B_log.md` | Yash |
| `Documentation_template.md` (final write-up, assembled from docs/method) | Mayank (at M5) |
| `make_submission.ps1` / `make_submission.sh`, `docs/REPRODUCE.md` | Yash |
| `docs/submissions_log.md` (leaderboard uploads) | Mayank |
| `CLAUDE.md` | shared-frozen; personal notes go in `CLAUDE.local.md` (gitignored) |
| `docs/plan/00_combined_plan.md` | shared-frozen |
| `docs/plan/01_*` / `02_*` | respective owner |

Parameters: each module keeps its own tunables as module-level constants (e.g. `blocking.TOPK_NAME`), **not** in `config.py`.

Enforced on GitHub by `.github/CODEOWNERS` (Phase 0): PRs touching the other person's files auto-request their review.

## 5. Git workflow
- **Two long-lived branches: `mayank` (Track A) and `yash` (Track B).** Each person commits only to their own branch and only touches files they own (section 4).
- **Merge into `main` between tasks** (each finished task / milestone step): push your branch, open a PR `mayank → main` or `yash → main`, tests green, merge with a **merge commit** (not squash — squashing a long-lived branch makes the next PR re-show old commits).
- **Right after every merge into `main`, both people sync:**
  ```bash
  git checkout <your-branch> && git fetch origin && git merge origin/main && git push
  ```
  Because file ownership is disjoint, these merges are conflict-free; a conflict means someone edited a file they don't own.
- Shared-file / contract changes: done by the file's owner (Mayank for Phase 0 files) on their branch, merged to `main` in a **PR of their own** that the other approves, then both sync immediately.
- Before opening a PR: `python -m pytest -q tests` on your branch after merging `origin/main`.
- Notebooks: clear outputs before committing (Kernel → Restart & Clear Output), keep them exploration-only; reusable code goes into `src/`.
- Never commit data, caches, models or full-size outputs (already gitignored). Share large artefacts via S3 or Drive.
- Plain commit messages (no tool attribution lines).

## 6. Milestones

| # | What | Mayank (A) | Yash (B) | Exit criterion |
|---|---|---|---|---|
| **M0** | Foundations | Phase-0 PR (`mayank → main`): `contracts.py`, `stub.py`, CODEOWNERS, requirements split, `CLAUDE.local.md` in .gitignore | Accept invite, create branch `yash` from `main`, set up env, build cache + sample, review Phase-0 PR | Both can run stub C1/C2 on the sample; tests green |
| **M1** | v1 components on the sample | EDA (A_01), `normalize.py` v1 → real C1; `blocking.py` v1 → real C2 for `sample_5pct` | `features.py` v1 + `train.py` + `predict.py` on **stub** C2; leave-one-country-out check | A: pair recall ≥ 0.90 at ≤ 60 cands/S1. B: pipeline produces valid-fold F0.5 on stub |
| **M2** | Integration on the sample | `run_pipeline.py --dataset sample_5pct` end to end | switch to real C2, retune threshold | End-to-end valid-fold F0.5 reported in both logs |
| **M3** | Full scale + first submission | Chunked full `train` + `test` blocking (AWS if RAM short); `run_pipeline --dataset test` | Train on full-train candidates (subsampled), final threshold; `make_submission` script | First leaderboard upload, validator PASS, logged |
| **M4** | Improve (loop) | Recall: transliteration dictionary, abbreviation mining, reverse blocking, France checks | Precision: context features, expected-F0.5 decision, error analysis, optional embeddings | Each change logged with valid-fold F0.5; only improvements merged |
| **M5** | Freeze + package | Assemble `Documentation_template.md`; final full run | Clean-clone reproduction test (`docs/REPRODUCE.md`); build zip | Zip reproduces the uploaded file byte-for-byte (or documented diff) |

Throughput rule: **A and B never wait on each other** — B uses stubs until M2, A measures blocking with `evaluate.blocking_report` (oracle F0.5) without needing the model.

## 7. Evaluation protocol (both tracks)
- Dev loop: `sample_5pct`, train on `fold == "train"`, report on `fold == "valid"` via `evaluate.score_report` / `blocking_report`.
- Every experiment gets one line in your own log (`docs/experiments/A_log.md` / `B_log.md`): date, commit, change, blocking recall/cands per S1 or F0.5 (overall, US, India, singleton).
- France proxy: train on one country, evaluate on the other (leave-one-country-out). A feature that helps in-country but hurts cross-country is suspect.
- Sample scores are optimistic (~12× fewer distractors). Before M3 sign-off, confirm on a full-train valid fold.

## 8. Integration points (the only moments we touch each other's output)
1. End of M0: stub C1/C2 on the sample exist on both machines.
2. M1 → M2: A announces "real C1/C2 for `sample_5pct` ready at commit X". B re-runs.
3. M3: A announces full `train`/`test` C1/C2 ready (shared via S3 if computed on AWS). B trains/scores.
4. Each leaderboard upload: Mayank uploads, logs in `docs/submissions_log.md` with both commit hashes.

## 9. Compute
- Local (16 GB) for everything on the sample and for model training on a subsample of pairs.
- AWS credits for full-scale blocking/feature building if local RAM is insufficient — see CLAUDE.md "Compute resources". Same region for both accounts; one shared S3 bucket for C1/C2/model artefacts.

## 10. Risks
| Risk | Mitigation |
|---|---|
| France behaves differently (unseen) | country-agnostic features; leave-one-country-out validation; eyeball test France C2 |
| Blocking recall caps the score | track oracle F0.5; reverse blocking; multiple independent blockers |
| Full-scale memory blow-up | chunk S1 by country and id bucket; float32; sparse top-K only |
| Contract drift | validators in `contracts.py`, tests on both sides |
| Last-minute packaging failure | M5 reproduction test from a clean clone well before the deadline |
