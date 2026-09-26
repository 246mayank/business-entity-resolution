# Track A — Mayank: Normalisation, Blocking, Orchestration

Branch: **`mayank`**. Read first: [00_combined_plan.md](00_combined_plan.md) (contracts C1/C2, file ownership, workflow) and [CLAUDE.md](../../CLAUDE.md).

**You own:** `src/normalize.py`, `src/blocking.py`, `src/run_pipeline.py`, `src/contracts.py` + `src/stub.py` (Phase 0 author), `tests/test_normalize.py`, `tests/test_blocking.py`, `tests/test_pipeline.py`, `notebooks/A_*.ipynb`, `requirements/blocking.txt`, `docs/method/blocking.md`, `docs/experiments/A_log.md`, `docs/submissions_log.md`, `Documentation_template.md` (at M5).
**Never edit:** `features.py`, `train.py`, `predict.py`, Yash's tests/notebooks/docs.

**Your north-star metrics** (from `evaluate.blocking_report` on `sample_5pct`, valid fold, then full train):
- `pair_recall` ≥ 0.95 (stretch 0.98) — true matches that survive blocking
- `candidates_per_s1` ≤ 60 (keeps Yash's feature/model cost bounded)
- `oracle_macro_f05` ≥ 0.97 — the ceiling you hand to the model
- Full test run fits the compute budget (local 16 GB, or a few AWS hours)

---

## A0 — Phase 0 (do first, unblocks Yash) · one PR `mayank → main`
1. **`src/contracts.py`**
   - Constants: `C1_COLUMNS = {"entity_id": pl.String, ..., "is_native_script": pl.Boolean}`, `C2_COLUMNS`, `C3_COLUMNS`; bit flags `FLAG_NAME=1, FLAG_ADDR=2, FLAG_POSTCODE=4, FLAG_REVERSE=8, FLAG_EXACT=16`.
   - Paths: `norm_path(split, source)`, `candidates_path(dataset)`, `features_path(dataset)`, `scores_path(dataset)` under `config.CACHE_DIR` (`dataset ∈ {"sample_5pct","train","test"}`).
   - Loaders: `load_norm(dataset, source)` (for `sample_5pct`, filter the train norm files to sample ids), `load_candidates(dataset)`.
   - Validators: `check_c1(df)`, `check_c2(df, s1_ids, other_ids)` (dtypes, no nulls, no duplicate pairs, ids exist, `s1_id` starts `S1-`, `id` starts `S2-`/`S3-`), `check_c3(df, c2)`, `check_c4(pairs, c2)` (subset of C2, each `id` at most once).
2. **`src/stub.py`** — `python -m src.stub` writes stub C1 (norm files restricted to the sample) and stub C2 for `sample_5pct` exactly as described in the combined plan §3. Must pass the validators. Keep it < 150 lines; it is throwaway.
3. **Repo plumbing**
   - `.github/CODEOWNERS` mapping files to `@246mayank` / `@Yashanand13` per the ownership table.
   - `requirements/blocking.txt`, `requirements/matching.txt` (empty with a comment), root `requirements.txt` becomes: shared pins + `-r requirements/blocking.txt` + `-r requirements/matching.txt`. Move `code/business_entity_resolution/requirements.txt` accordingly (keep it at that path — the submission needs it there; the `requirements/` folder sits next to it).
   - `.gitignore`: add `CLAUDE.local.md`.
   - Empty `docs/experiments/A_log.md`, `docs/experiments/B_log.md`, `docs/submissions_log.md` with a header table.
4. Tests: `tests/test_contracts.py` (validators reject a bad frame, accept the stub).
5. PR, Yash approves, merge, both sync. **Tell Yash “stub ready”.**

## A1 — EDA notebook `notebooks/A_01_text_eda.ipynb` ✅ done 2026-09-26 — findings in `docs/method/blocking.md`
Answer with numbers on the sample, write conclusions into `docs/method/blocking.md`:
- Script mix: % of names/addresses per Unicode script (Devanagari, Kannada, Tamil, Telugu, Bengali, Gujarati, Gurmukhi, Malayalam, Odia, Latin) per source. Which sources carry native script? (Example: S2 `राम मार्केटिंग प्राइवेट लिमिटेड`.)
- Junk: most frequent leading/trailing non-alnum sequences (`--`, `<<`, …), domain-like names (`\.(com|in|net|org|fr)\b`), words like “DBA”, “the”.
- Legal forms: top 100 first/last tokens of names; how often S1 and its match disagree on legal form or its position.
- Address anatomy per country: postcode presence & format (US 5 / 5+4, India 6 or `560 001`, France 5), house-number patterns (`#12`, `No. 183`, `12/3`, `KH NO. -570/13`, `19 1/2`, `5 bis`), state names vs codes, `near …` landmark phrases, component order.
- **On GT pairs** (the key part): how often match & S1 share postcode / house_no / first name token / any name token; distribution of rapidfuzz `token_set_ratio` on names and addresses for matches vs random same-country pairs. This tells you which blocking keys are safe.
- **Test France**: 200-row eyeball of test S1/S2/S3 France records — legal forms (SARL, SAS, SASU, EURL, SA, SCI, SNC), street words (rue, avenue, boulevard, bd, chemin, allée, place, bis/ter), postcode position. Note anything train didn’t prepare us for.

## A2 — `src/normalize.py` v1 → real C1 ✅ v1 done 2026-09-26 (see `docs/method/blocking.md` §2)
> **Revised after A1 EDA** — see `docs/method/blocking.md` §2: postcodes are almost never present (5-digit numbers are house numbers); add DBA/domain/bracket/honorific handling, a learned Indic token dictionary, and extra columns `name_nospace`, `addr_street`, `addr_city`, `addr_region`, `is_domain`.

Public API: `normalize_frame(df: pl.DataFrame) -> pl.DataFrame` (C1 columns), `normalize_all(splits=("train","test"))` writing `cache/norm/…`, CLI `python -m src.normalize`.
Pipeline per field (vectorised with Polars string ops wherever possible; Python UDFs only for transliteration, and only on rows with non-Latin chars):
1. Unicode NFKC; detect non-Latin → `is_native_script`.
2. **Transliteration** of Indic scripts to Latin: `indic_transliteration.sanscript` (script → ITRANS/IAST) then fold to ASCII and simplify (lowercase, strip diacritics, collapse doubled vowels, drop inherent final `a` where it helps — tune on GT pairs). Cache per unique string (many duplicates).
   - v2 idea (M4): **learn a token dictionary from train GT pairs** where one side is native and the other Latin (align tokens by position/similarity), e.g. `प्राइवेट → private`, `लिमिटेड → limited`, `ಕರ್ನಾಟಕ → karnataka`. Training data only — allowed.
3. Accent fold (`unidecode` or NFKD + drop combining marks), lowercase, `&` → ` and `, `@` → ` at `, punctuation → space, collapse spaces, strip leading junk tokens.
4. **Abbreviation expansion** (names + addresses): `corp→corporation`, `co→company`, `pvt→private`, `ltd→limited`, `intl→international`, `mfg→manufacturing`, `svc/svcs→services`, `st→street` (address only, not `st.` as saint at start), `rd→road`, `ave→avenue`, `blvd→boulevard`, `dr→drive`, `ln→lane`, `ct→court`, `hwy→highway`, `apt/unit/ste`, `bd→boulevard` (FR), `av→avenue`, `n/s/e/w` directions, `nagar`, `marg`… Start from a hand list, then **mine more from GT pairs**: token pairs (a, b) that co-occur aligned in matched names/addresses far more than chance where one is a prefix/abbrev of the other.
5. Legal form: detect anywhere in the token list (front or back), map to canonical (`private limited`/`pvt ltd`→`pvt_ltd`, `incorporated`/`inc`→`inc`, `l l c`/`llc`, `limited`/`ltd`, `llp`, `corporation`/`corp`, `company`/`co`, `sarl`, `sas`, `sasu`, `eurl`, `sa`, `sci`, `snc`, `gmbh`…). Remove from `name_core`.
6. `postcode`: per country regex on the raw address (US `\b\d{5}(-\d{4})?\b` keep 5; India `\b\d{3}\s?\d{3}\b`; France `\b\d{5}\b`) — but keep the extractor country-agnostic by design (try all patterns, prefer the one that matches the country’s dominant length learned from data, not hard-coded names).
7. `house_no`: first token matching `^\d+[a-z]?(/\d+)?$` or after `no|#|door no|plot no|kh no`.
8. US state names ↔ codes: map to codes (list mined from train pairs; ~60 values). Indian states: fold spelling variants.
Tests: table-driven cases from real examples (the ones in CLAUDE.md), plus property tests (idempotent, never raises on empty strings, preserves row count & ids). Performance target: full train+test (~23M rows) in < 15 min locally.

## A3 — `src/blocking.py` v1 → real C2
> **Revised after A1 EDA**: drop the postcode key; blockers = name char-TF-IDF on `name_nospace`, address TF-IDF, name+address TF-IDF, `(house_no, street token)` / exact-name keys, reverse top-3. See `docs/method/blocking.md` §2.

Public API: `build_candidates(dataset: str) -> None` writes C2 and returns stats; CLI `python -m src.blocking --dataset sample_5pct`.
Work **per country**, S1 in chunks (e.g. 50k), S2 and S3 treated as one target pool but with a per-source top-K so neither source crowds out the other.
Blockers (union; each sets its bit in `blk_flags`):
1. **Name TF-IDF**: `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,4), min_df=2, dtype=float32, sublinear_tf=True)` fitted on the country’s S1 ∪ S2 ∪ S3 `name_core`. Top-K by cosine via chunked sparse matmul — use `sparse_dot_topn` (MIT) or implement chunk × matrix then `argpartition`. K_name ≈ 15 per source.
2. **Name+address TF-IDF**: same on `name_core + " " + addr_clean` (catches generic names disambiguated by address). K ≈ 15 per source.
3. **Exact keys**: `(postcode, first name_core token)`, `(postcode, house_no)`, `(name_core)` exact. Cap each key group (skip keys with > 200 records — they are stopword-like).
4. **Reverse direction** (important because each record has exactly one owner): for each S2/S3 record retrieve top-k (k≈3) S1s by name+address TF-IDF; add those pairs; fill `blk_rev_rank`.
5. Merge, dedupe, keep best sims/ranks; **cap per S1** at `MAX_CANDS` (≈ 60) by best combined score.
Record `blk_name_sim`, `blk_addr_sim` (compute for every final pair, even if it came from a key blocker — cheap row-wise cosine of the TF-IDF rows).
Measure with `evaluate.blocking_report` per blocker (ablation table in `A_log.md`) — which blocker contributes unique true pairs, what each costs in candidates.

Empty addresses (~3.4% of S2/S3): they can only be found by name blockers — check their recall separately.

## A4 — `src/run_pipeline.py`
`python -m src.run_pipeline --dataset {sample_5pct|train|test} [--steps normalize,block,features,train,predict,write]`
Calls public functions only: `normalize.normalize_all`, `blocking.build_candidates`, `features.build_features`, `train.fit`, `predict.score`, `predict.decide`. Writes `output/candidate_pairs.tsv` (from C2) and `output/matching_results.tsv` (from C4) with `io_utils.write_id_lists`, over **all** S1 ids of the dataset; for `test` runs the organisers' validator at the end. For `sample_5pct`/`train` prints `evaluate.score_report` on the valid fold. Until Yash’s modules exist, the features/train/predict steps can be skipped with `--steps`.

## A5 — Full scale (M3)
- Memory plan: per country, per S1 chunk; TF-IDF matrices float32 CSR; target pool per country ≤ ~4M rows (India test ≈ 4.4M S2+S3 — verify). Profile peak RSS on train India first.
- If > ~12 GB: run on an AWS memory-optimised instance (see CLAUDE.md Compute): clone repo, `aws s3 cp` the dataset zip, run `python -m src.run_pipeline --dataset test --steps normalize,block`, upload `cache/norm` + `cache/candidates` to the shared S3 bucket, stop the instance. Write the exact commands to `docs/method/blocking.md` (appendix “AWS runbook”).
- Validate test C2 with `contracts.check_c2`; report candidates/S1 per country, incl. France (no GT — compare distribution with US/India).

## A6 — Improve recall (M4 loop)
In priority order, each measured on the full-train valid fold:
1. Learned transliteration dictionary + abbreviation mining (A2 v2 ideas).
2. Phonetic key blocker on `name_core` (e.g. Double Metaphone of first two tokens — `jellyfish` MIT, or a simple consonant skeleton) for typos like `Tetlecommunication`.
3. Token-order-insensitive key: sorted tokens of `name_core`.
4. Tune K per country size; adaptive K (take more when top scores are close).
5. France sanity: synthetic check — does the pipeline treat `5 bis Rue …` / `SARL` sensibly? (No labels, so inspect.)

## A7 — Packaging (M5)
- Assemble `Documentation_template.md` from `docs/method/blocking.md` + Yash’s `docs/method/matching.md` + results table.
- Final full run, upload, log in `docs/submissions_log.md`.

## Hand-offs you produce
| When | What | Tell Yash |
|---|---|---|
| end A0 | stub C1/C2 (`python -m src.stub`) | “stub ready, merged at <commit>” |
| end A2+A3 on sample | real C1/C2 for `sample_5pct` | “real sample C2 at <commit>, recall X, cands/S1 Y” |
| A5 | full `train` + `test` C1/C2 (local path or S3 URL) | where to get it + stats |

## Suggested order / parallelism
A0 → (A1 time-boxed) → A2 v1 → A3 v1 → hand-off → A4 → A5 → A6 loop → A7.
