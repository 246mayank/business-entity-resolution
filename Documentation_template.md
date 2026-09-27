# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** BTards
**Team Members:** Mayank Singh (text normalisation, candidate generation, orchestration), Yash Anand (features, matching model, decision)
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

The pipeline has three stages:

1. **Normalisation with learned dictionaries.** Indic-script → English and state-code → state-name maps are mined only from the training ground truth.
2. **Scalable, country-partitioned sparse TF-IDF retrieval** followed by a **learned pruning stage**. Candidates per Source 1 entity drop from **~50 to ~5–6** with **no loss in final F0.5**.
3. A **LightGBM matcher** over 39 string, address-number and candidate-context features. It is followed by a **one-owner assignment**, since each Source 2/3 record belongs to at most one Source 1 entity, and a threshold tuned directly on macro F0.5.

Every learned component is fitted on train only. Nothing depends on the country label, so France, which never appears in training, is handled the same way as US and India.

Headline numbers:
- full-train validation macro F0.5 **0.9415** (first model)
- public leaderboard **0.935** (first submission)
- final model (pruned candidates + address-number features): full-train validation macro F0.5 **0.9579**

---

## 2. Methodology

### 2.1 Problem Analysis

EDA was done on a deterministic 5% dev sample plus the full data (`notebooks/A_01_text_eda.ipynb`, `docs/method/blocking.md`). The findings that shaped the design:

**Scale and structure**
- Train: 2.2M S1, 10.3M S2+S3. Test: 1.73M S1, 10.0M S2+S3.
- 5.6% of train S1 are singletons.
- Every S2/S3 record belongs to **at most one** S1 (0 of 7.64M matched ids have two owners).
- Matched pairs **always share `country`**.

**Scripts**
- Train S1 and all US records are ASCII.
- ~23% of S2 India names and ~24% of addresses contain Indic script: Devanagari 13%, plus Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Odia and Gurmukhi.
- The names are phonetic renderings of English words (`कंसल्टेंसी` = consultancy, `प्राइवेट लिमिटेड` = private limited).
- In addresses, only the state name is native (`…, PUNE, महाराष्ट्र`).

**Name noise (S2/S3 only; S1 is clean)**
- leading junk (`>>`, `--`, `***`, `#`, `@`)
- tokens wrapped in `[]` / `()`
- names squashed into web domains (`downtownautobody.com`, 4.5%)
- random alias + `DBA` (`Lyraxylo Co DBA Harbor Esports L.L.C.`)
- honorifics (`M/s`, `Smt`, `Shri`, `The`)
- legal forms moved to the front
- digit-for-letter typos (`Precisi0n`)
- injected accents (`Prívate`)
- names **replaced by made-up words** (`Atlantic Center PLLC` → `Zetazeph`, with the address intact)

**Address noise**
- **Postcodes are essentially absent**: <1% Indian PINs, and US 5-digit numbers are house numbers. Postcode blocking is useless.
- Component order is shuffled.
- `NULL` / `<NULL>` / `N/A` placeholders.
- Empty addresses: 3–4% of S2/S3.
- State level differs between sources: US codes vs full names; France region in S1 vs *département* in S2/S3.
- Landmarks (`near`, `opp`) appear in India.
- **Digits are perturbed**, e.g. `1919` vs `1920`.

**What true matches share, vs random same-country pairs**

| Evidence | True matches | Random pairs |
|---|---|---|
| Share a name token | 86% | 18% |
| Share an address token | 95% | 14% |
| Equal first number | 79% | 0.5% |

- **14% of true matches share no name token** (half native-script, the rest domain names, made-up names or heavy typos). For these, the address and space-less character n-grams are the only route.
- Test contains **France (15% of test S1)**: French legal forms (`SARL`, `SAS`, `EURL`, `SASU`, `SCI`), the abbreviation pair `ets`/`etablissements`, and street words (`Rue`, `Bd`, `Imp.`, `bis`/`ter`).

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned candidate pruning + gradient-boosted pairwise classifier + global one-owner decision (hybrid).

**Core innovations:**
1. **Learned, leakage-free normalisation dictionaries** (Indic tokens and a per-country place map) mined only from train GT pairs.
2. **Place-conditioned sparse retrieval.** The combined name+address search runs inside each shared address component (city, district or state), which separates generic names ("… Private Limited") that global search cannot.
3. **Two-stage candidate generation.** Retrieval gives ~50 candidates; a 17-feature learned pruner cuts that to ~5 per S1. The matcher only ever scores the pruned set, which is what `candidate_pairs.tsv` contains.
4. **One-owner assignment and a decision rule tuned directly on macro F0.5**, including singletons, with a precision guard.

```
raw TSV ──► Parquet cache ──► normalize (C1) ──► blocking: 6 sparse TF-IDF searches (C2, ~50/S1)
        ──► features (39) ──► learned pruner (≤10/S1, p≥0.01 → ~5/S1) = candidate_pairs.tsv
        ──► LightGBM matcher ──► threshold + one-owner ──► matching_results.tsv
```

All stages exchange validated Parquet files (`src/contracts.py`: schemas, id-existence, same-country and one-owner checks). Every step is chunked or streamed so the full pipeline runs on a 16 GB laptop.

**Validation protocol**
- S1 entities are split deterministically by numeric id into a train fold (80%) and a valid fold (20%).
- The matcher early-stops on a separate 10% of the train fold, so the valid fold is only used for threshold tuning and reporting.
- Validation tracked the leaderboard closely: **0.9415 valid vs 0.935 public**.

---

## 3. Candidate Generation (Blocking)

### 3.1 Normalisation (`src/normalize.py`)

All steps are vectorised Polars. Transliteration runs once per unique string.

- **Unicode.** NFKC; zero-width and control characters removed; accent folding (NFKD, combining marks dropped).
- **Indic → English.**
  - *Learned dictionary.* For every train GT pair whose S2/S3 side is native script, each native token is aligned to the S1 token with the most similar phonetic skeleton (unidecode, lowercase, collapsed repeats; Jaro-Winkler ≥ 0.6).
  - *Acceptance.* An entry is kept with ≥ 2 votes and ≥ 60% agreement. The result is 1,236 entries covering ~91% of native token occurrences (`लिमिटेड`→limited in 9 scripts, `कंसल्टिंग`→consulting).
  - *Fallback.* Unknown tokens get the phonetic unidecode form.
- **Names.**
  - `X DBA Y` → name `y`, with `x` kept as the alias.
  - Web domains unwrapped (`www.`, TLD); `M/s` removed.
  - `&`→and; dots and apostrophes dropped; `0`→`o` inside words; abbreviation map.
  - Legal forms are stripped from the ends; unambiguous ones (`inc llc llp ltd pvt corp plc sarl sasu eurl gmbh`) are also stripped anywhere. They are collected into a canonical `legal_form` (`ltd_pvt`, `llc`, `sarl`, …).
  - Honorifics removed.
  - Outputs: `name_core`, and `name_nospace` for squashed names.
- **Addresses.**
  - Null placeholders removed; abbreviations (US/India/France); US state names → codes; `city` suffix dropped; leading zeros stripped.
  - Outputs: `house_no` (first number-like token: `59a`, `570/13`, `8-2-67/1`), `addr_street` (the component holding it) and `addr_places` (digit-free components, compared as a set).
- **Learned per-country place map.** An S2/S3 component that S1 of the same country almost never uses (< 5% of its S2/S3 frequency) is linked to the S1 component it co-occurs with in ≥ 80% of matched pairs.
  - It learned 94 variants: all Indian state codes (`mh`→maharashtra, `tn`→tamil nadu), transliteration leftovers (`pshcimbng`→west bengal), and city aliases (`bombay`→mumbai).
  - The target is **appended**, never replacing the variant, and only within its own country (`tn` stays Tennessee in the US).
  - Share of India S2/S3 records with a canonical state name: **97%**.
- **Effect on true pairs:**

  | | Stub normalisation | normalize v1 |
  |---|---|---|
  | Share a name token (native-script pairs) | 0.52 | **0.94** |
  | Share a name token (all pairs) | 0.86 | 0.91 |
  | Address token-set ratio p10 | 74 | 83 |

- **Runtime:** full train + test (23M records) in ~4 min, one process per split.

### 3.2 Sparse retrieval (`src/blocking.py`)

The search runs per country (never across countries), with S2 and S3 as one target pool. Each search is a sparse TF-IDF cosine top-K over the pool (`sparse_dot_topn`). An **absolute** document-frequency cap (`MAX_DF = 3000`) keeps the cost per query independent of data size.

| Blocker (flag) | Representation | Top-K | Recall alone* | True pairs only it finds* |
|---|---|---|---|---|
| name (1) | `name_core` word 1–2-grams | 10 per source | 0.598 | 5,250 |
| address (2) | `addr_clean` words | 10 per source | 0.599 | 1,148 |
| combined + place-conditioned (16) | name ⊕ address; also searched within every shared `addr_places` value | 10 per source / 10 per place | **0.917** | **53,994** |
| char 4-grams (4) | `name_nospace`, only against single-token / domain names | 5 | 0.051 | 14,599 |
| reverse (8) | each S2/S3 record → its top-3 S1 | 3 | 0.867 | 5,500 |

\* Full train, valid fold.

- Pairs are merged, scored (`blk_name_sim` and `blk_addr_sim` cosines), ranked, and capped at 60 per S1. A record's best S1 is kept up to rank 120.
- **Wide candidate set:**
  - Train: 108.3M pairs, **49.1 per S1**, pair recall **0.945**, oracle macro F0.5 **0.979**, 70 min.
  - Test: 86.8M pairs, **50.1 per S1** (France 54, India 49.6, US 49.2), 52 min.

### 3.3 Learned candidate pruning (`src/prune.py`) — the final candidate stage

A LightGBM model (200 trees, 63 leaves) is trained on the train fold over 17 cheap signals:
- blocking similarities, ranks and flag bits
- candidate counts per S1 and per record
- rank gap
- name / address token-set ratio
- space-less name ratio
- place Jaccard
- house-number match

It keeps **at most 10 candidates per S1 with pruner probability ≥ 0.01**. The full 39 features are then rebuilt on the kept pairs only, so candidate-context features (`n_cands`, `n_s1_for_id`, rank gap) describe the final set. The matcher is trained on, and predicts only, these pairs, and they are exactly `output/candidate_pairs.tsv`.

| Full train, valid fold | Wide (retrieval) | **Pruned (final)** |
|---|---|---|
| Candidates per S1 | 49.1 | **5.11** |
| Pair recall | 0.9463 | 0.9436 |
| Oracle macro F0.5 | 0.9801 | 0.9791 |
| Matcher macro F0.5 (same model) | 0.9431 | **0.9430** |

- **Test:** 86.8M → **10.7M pairs, 6.18 candidates per S1**. 2,524 S1 have no candidate, so they get empty lists.
- **Candidate pairs in the submitted file:** **10,706,441** for 1,732,544 test S1 (6.18 per S1).

**How true matches are protected**
- Six complementary retrieval routes, each contributing unique true pairs (table above).
- The reverse search exploits the one-owner structure.
- Place-conditioned search handles generic names.
- The char 4-gram search handles squashed and domain names.
- The pruner is tuned to lose < 0.3% of true pairs.
- Recall and the oracle ceiling are tracked on the full-train valid fold for every change (`docs/experiments/A_log.md`).

---

## 4. Matching Model

**Features** (39; `src/features.py`, computed on the pruned candidates in 20k-S1 chunks and streamed to disk)

- **Name**
  - rapidfuzz token-set, token-sort, partial and plain ratio on `name_core`
  - ratio and partial ratio on the space-less name (domain-squashed names)
  - char 2–3-gram TF-IDF cosine (vectoriser fitted once on train and persisted)
  - length ratio
  - name frequency of both sides among the dataset's S1 (common names are weak evidence)
  - legal-form agreement: 3-state (equal / missing / conflict)
- **Address**
  - token-set and partial ratio; TF-IDF cosine
  - empty-address flag
  - place Jaccard over `addr_places`
  - house-number agreement (3-state), edit distance and log-gap
  - **number-set features:** both have numbers, number Jaccard, and the minimum edit distance and minimum log-gap between unmatched numbers. These separate digit typos (748 vs 548) from neighbouring buildings (2579 vs 2582).
  - a combined name/address score
- **Candidate context**
  - blocking cosines, rank and reverse rank
  - the five blocker flag bits
  - `n_cands` per S1
  - `n_s1_for_id` (S1s competing for the same record, computed over the full candidate set)
  - number of near-identical-name candidates
  - rank gap to the best candidate
- **Source**
  - `is_s3`; native-script flag of the S2/S3 record
- **Country is never a feature**, so France is not out-of-vocabulary.

**Model type:** LightGBM binary classifier (MIT licence; tree ensemble, far below 8B parameters).
- Parameters: `num_leaves` 127, `learning_rate` 0.05, feature/bagging fraction 0.8, L1 0.1 / L2 1.0, `scale_pos_weight = neg/pos`, up to 3000 rounds.
- Early stopping on a held-out 10% of train-fold S1.
- Final model: trained on the **pruned** candidates of 20% of train S1 (1.57M pairs, 64% positives, the hard cases near the boundary); early stopping picked 2,740 of max 6,000 rounds; threshold 0.62 (no precision guard). Trained in 2.5 min on an 8-vCPU AWS instance.

**Decision.**
- Score → keep pairs with `p ≥ t` → **one-owner assignment** (each S2/S3 record goes only to its highest-probability S1).
- `t` is grid-searched on the valid fold to maximise **macro F0.5 including singletons**, subject to micro precision ≥ 0.987.
- The chosen `t` = 0.97; the unguarded optimum was 0.96, a difference of 0.0011.
- The threshold and rule are persisted in `models/decision.json`. The pipeline refuses to apply a decision tuned on a different dataset than the model.

---

## 5. Results & Error Analysis

| Stage / metric | Value |
|---|---|
| Candidates per S1, test (wide → pruned) | 50.1 → **6.18** |
| Candidate recall / oracle F0.5, train valid (pruned) | 0.9436 / 0.9791 |
| **Macro F0.5, full-train valid fold (first model)** | **0.9415** (US 0.9473, India 0.9329) |
| Micro precision / recall | 0.9889 / 0.8735 |
| Singleton / non-singleton F0.5 | 0.9559 / 0.9406 |
| Public leaderboard, first submission | **0.935** |
| **Final model — valid macro F0.5** | **0.9579** (US 0.9607, India 0.9538) |
| Final model — micro precision / recall | 0.9906 / 0.9096 |
| Final model — singleton / non-singleton F0.5 | 0.9681 / 0.9573 |
| Final model — S1 with ≥ 1 false match (valid) | 2,570 (first model 2,935) |
| Final model — public leaderboard | [pending upload] |

Test output (first model): 5,374,874 matches, about 3.1 per S1, the same as train. Empty lists: France 6.6%, India 7.2%, US 6.5%. The unlabelled country behaves like the labelled ones.

**Common false positives (wrong merges).** 2,993 pairs on 88k valid S1; 229 on true singletons.
- Mostly **the same or near-identical business name at a neighbouring address**: `1390` vs `1393 Road V`, flat `101` vs `104` in the same society, unit `a5` vs `a8`.
- 45% have an identical core name, and 37% have *conflicting* house numbers. These are sister branches or chains.
- Slightly more in India (51% of FPs).

**Common false negatives (missed matches).** 38,628 pairs.
- **43% never reach the candidate list** (the blocking ceiling). They are dominated by heavy transliteration garbage and records with no shared name and a sparse address.
- Of the 22,055 the model rejects, **55% have conflicting house numbers** caused by digit noise (`1919` vs `1920`, `16` vs `18`), which is the mirror image of the false-positive pattern.
- 18% have an empty S2/S3 address.
- Some residual name noise (`1imited` with a digit-for-letter typo; unaligned transliterations like `priiabthte`).
- The address-number features were designed from exactly this analysis. On the dev sample they moved F0.5 from 0.9808 to 0.9851 and cut entities with a false match from 558 to 436.

---

## 6. Conclusion

Most of the gain came from:
- normalisation dictionaries learned from the training labels
- **multi-route, country-partitioned sparse retrieval**
- a **learned pruning stage** that shrinks candidates ~10× at no cost in F0.5
- a precision-oriented LightGBM matcher with a global one-owner decision tuned on the competition metric

The main lesson: the 5% sample over-estimated every score (12× fewer distractors), while a full-train valid fold predicted the leaderboard within 0.007. Measuring at full density early drove the place-conditioned search, the pruning stage and the streaming implementation that fits the pipeline in 16 GB.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

| Module | Role |
|---|---|
| `src/io_utils.py` | TSV reading (all strings, no quoting/NA inference), Parquet cache, submission writers |
| `src/sample.py`, `src/config.py` | deterministic id-bucketed dev sample and valid fold; paths |
| `src/contracts.py` | stage contracts C1–C4 (schemas, validators, loaders) |
| `src/normalize.py` | normalisation, learned Indic dictionary and place map |
| `src/blocking.py` | sparse TF-IDF retrieval (6 searches) |
| `src/features.py` | 39 pair features (chunked, streamed) |
| `src/prune.py` | learned candidate pruning (final candidate stage) |
| `src/train.py`, `src/predict.py` | LightGBM matcher; scoring, threshold tuning, one-owner decision |
| `src/evaluate.py` | macro F0.5 (per entity, singletons included), blocking recall and oracle |
| `src/run_pipeline.py` | end-to-end orchestration |
| `tests/` | 67 unit tests |

End to end, from `code/business_entity_resolution/` (exact commands and timings in its README):

```bash
python -m src.io_utils && python -m src.normalize
python -m src.blocking --dataset train && python -m src.blocking --dataset test
python -m src.run_pipeline --dataset train --steps features prune train predict write --s1-pct 20
python -m src.run_pipeline --dataset test --steps features prune predict write validate --model-from train
```

**External data:** none. All dictionaries and maps are learned from the provided training data. No APIs, geocoders or pretrained language models are used.

### B. Additional Results
- Blocking ablation and full-density evidence: `docs/method/blocking.md`.
- Experiment logs: `docs/experiments/A_log.md`, `docs/experiments/B_log.md`.
- Submission log: `docs/submissions_log.md`.
