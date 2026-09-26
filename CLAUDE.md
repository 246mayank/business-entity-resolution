# Business Entity Resolution — ML Challenge 2026

Match Source 2 / Source 3 business records to deduplicated Source 1 entities. Full problem statement: `README.md`. Methodology write-up: `Documentation_template.md` (fill in, ships in the zip).

## Hard rules (violating these = rejection or disqualification)
- **No external data**: no APIs, geocoders, business registries, web lookups, or downloaded gazetteers. The pipeline reads only `dataset/`. Pretrained model weights are OK only if MIT/Apache-2.0 and ≤ 8B params — record name + license in `requirements.txt`/docs.
- **Country is an open set**: test adds `France` (absent from train). Never hard-code, filter, or one-hot `{US, India}`. Features must be country-agnostic; country may only be used as a blocking partition / string-equality feature.
- **Output format** (both files tab-separated, UTF-8, no quoting, no index):
  - `output/matching_results.tsv` — header `source1_entity_id<TAB>matched_entity_ids`
  - `output/candidate_pairs.tsv` — header `source1_entity_id<TAB>candidate_entity_ids`
  - Exactly one row per test S1 entity (1,732,544), empty list for no match, IDs comma-joined, only S2-/S3- IDs from the test set, no duplicates.
  - Matches ⊆ candidates. `candidate_pairs.tsv` = the exact set the final model scores (last filtering stage), not an earlier blocking pass.
- **Always validate** before any upload:
  `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` (add `--check-ids` for final).
- **Reproducibility**: `code/business_entity_resolution/` must regenerate both outputs from raw data alone (README with exact commands, pinned `requirements.txt`, fixed seeds). Relative paths via `src/config.py`; nothing AWS- or machine-specific.

## Metric
Macro F0.5 per S1 entity, averaged over **all** S1 entities (singletons included: empty prediction on a singleton = 1.0, any prediction = 0.0). Precision weighs 2× recall → when unsure, predict fewer matches. Tune thresholds on this exact metric, never on accuracy/AUC or the 0.5 default.

## Data facts (verified)
| file | rows |
|---|---|
| train S1 / S2 / S3 | 2,206,821 / 5,034,616 / 5,285,603 |
| train ground truth | 2,206,821 rows; 123,247 singletons (5.6%); 7,638,365 matched IDs |
| test S1 / S2 / S3 | 1,732,544 / 4,887,273 / 5,082,316 |

- Columns: `entity_id, business_name, business_address, country`. Source = ID prefix.
- ~74% of S2/S3 records match some S1; ~3.7 matches per non-singleton S1 (≈1.8 from S2 + 1.9 from S3), max 11.
- **Verified: every S2/S3 record belongs to at most one S1** (0 of 7.64M matched ids have 2 owners) → enforce one-owner assignment at prediction time.
- **Verified: matched pairs always share the same `country`** (0 of 7.64M differ) → blocking within country loses nothing.
- All GT ids exist in the train sources; no empty names or countries; ~3.4% of S2/S3 addresses are empty (S1 never).
- Train S1: US 1.32M, India 0.88M. **Test S1: India 0.81M, US 0.66M, France 0.26M (15%)** — France handling matters.
- Exact normalised-name match within country (5% sample, valid fold): precision 0.46, recall 0.20, macro F0.5 0.364 → names alone are ambiguous (generic names repeat); address evidence is essential.
- Noise seen: Devanagari/Kannada script names & addresses; junk prefixes (`--`, `<<`); legal suffix moved to front (`Pvt. X Ltd.`, `LLC X`); typos (`Tetlecommunication`); domain-as-name (`foo.com`); UPPERCASE abbreviated S2 addresses; state name vs code (Texas/TX); reordered address components; empty addresses; accents; French forms in test (`Rue`, `bis`, `SARL`).

## Engineering rules
- Machine: 15.7 GB RAM local; AWS (Builder Center) available for full-scale runs — code must run on both.
- Read TSVs only through `io_utils.read_tsv` (all strings, empty stays "", quoting disabled). Parquet cache already built by `python -m src.io_utils`; downstream code reads Parquet via `io_utils.load_source` / `load_gt`.
- Write submission files only through `io_utils.write_id_lists`; score with `src.evaluate` (`score_report`, `blocking_report`). Dev sample: `sample.load_sample()`, validate on `fold == "valid"`.
- Run Python via the repo `.venv`; set `PYTHONIOENCODING=utf-8` on Windows consoles (Indic text).
- Never materialise a full cross product. Block within country, process S1 in chunks, use sparse top-K retrieval + exact keys.
- Develop on a fixed-seed sample (~5% of train S1 + their GT matches + random S2/S3 noise); run full data only for final numbers.
- Validation split is **by S1 entity** (GroupShuffleSplit/GroupKFold) — never a random pair-level split.
- Track two numbers every iteration: blocking recall (GT matches present in candidates) + avg candidates per S1, and end-to-end macro F0.5.
- Fit vocabularies/maps (TF-IDF, abbreviation tables) on train only; apply to test.
- No `GridSearchCV` on full data, no dense `.toarray()` on big sparse matrices, no `StandardScaler` for tree models.
- `cache/`, `models/`, `dataset/` are working dirs — not part of the submission zip.

## Planned pipeline (src/)
1. `io_utils.py` — TSV→Parquet, TSV writers.
2. `normalize.py` — lowercase, NFKD accent strip, Indic→Latin transliteration, junk punctuation removal, abbreviation expansion, legal-suffix extraction, parse postal code / house number / city / state.
3. `blocking.py` — union of char n-gram TF-IDF top-K (name, name+address) and exact keys (postal code, house no. + name token); writes `candidate_pairs.tsv`.
4. `features.py` — rapidfuzz similarities, TF-IDF cosine, postal/number/city agreement, legal-suffix agreement, candidate rank & score gap; optional small multilingual embedding (MIT/Apache, e.g. multilingual-e5-small).
5. `train.py` — LightGBM on pair features.
6. `predict.py` — score, per-entity threshold tuned for macro F0.5, one-owner assignment of S2/S3 records, write `matching_results.tsv`.
7. `evaluate.py` — macro F0.5 on held-out S1 entities.
8. `run_pipeline.py` — end-to-end entry point.

## Submission zip
`<team_name>_submission.zip` = `output/` (both TSVs) + `code/business_entity_resolution/` (`src/`, `README.md`, `requirements.txt`) + `Documentation_template.md`. Built by `make_submission.ps1`.

## Collaboration / setup
- Git repo root = `student_resource/`. `dataset/`, `cache/`, `models/`, `*.zip`, and full-size `output/*.tsv` are gitignored (GitHub 100 MB file limit; competition data stays out of the repo).
- New collaborator: clone, then unzip the organisers' `student_resource.zip` and copy its `dataset/` folder into the repo root.
- Work on feature branches, merge to `main` via PR; keep `main` runnable.

## Compute resources (from organisers' AWS PDF)
- Each team member gets their own AWS Free Tier account: up to $200 credits (6 months). SageMaker 2-month trial = 250 h `ml.t3.medium` notebook (2 vCPU / 4 GB — too small for this data) + 50 h `ml.m4/m5.xlarge` training (4 vCPU / 16 GB ≈ our laptop). S3 free: 5 GB, 20k GET, 2k PUT / month.
- Top-500 leaderboard teams get extra credit codes.
- Plan: develop + train locally (16 GB). Use AWS only for full-scale blocking/inference if local RAM is not enough — a memory-optimised EC2/SageMaker instance paid from credits, same region for every member, stop it after each run, set a billing alert.
- Share large files between members via one S3 bucket (bucket policy or `aws s3 presign`); never commit them to git.
