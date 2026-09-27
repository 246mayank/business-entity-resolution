# Track B method notes: features, matching model, decision

Owner: Yash. Code: `src/features.py`, `src/train.py`, `src/predict.py`. Experiment rows: `docs/experiments/B_log.md`.

Input is the pruned candidate set (`src/prune.py`, ~5 per S1). Output is `models/matcher.txt` plus
`models/decision.json`, and the final pairs that `run_pipeline` writes to `matching_results.tsv`.

## 1. Features (46 = the 39 below + 7 in §2b)

Computed per chunk of 20k S1 entities and streamed to one Parquet file, so peak memory stays
at 2.5–3 GB regardless of dataset size. Strings come from normalisation (C1): `name_core`,
`addr_clean`, `house_no`, `legal_form`, `addr_places`.

| Group | Features | Why |
|---|---|---|
| Name similarity (7) | rapidfuzz `token_set_ratio`, `token_sort_ratio`, `partial_ratio`, `ratio` on `name_core`; `ratio` and `partial_ratio` with spaces removed; char 2–3-gram TF-IDF cosine | Different noise types: reordering, truncation, typos, squashed domain names |
| Name context (4) | length ratio; log count of S1 sharing each side's exact name (`name_freq_s1`, `name_freq_id`); legal-form agreement (1 equal / 0 missing / −1 conflict) | A common name ("all media") is weak evidence; a conflicting legal form is strong counter-evidence |
| Address similarity (5) | `token_set_ratio`, `partial_ratio`, TF-IDF cosine; empty-address flag; Jaccard of `addr_places` | Component order is shuffled; places are digit-free city/district/state sets |
| House number (3) | agreement (3-state), edit distance, log numeric gap | See §2 |
| Address numbers (4) | both have numbers; Jaccard of all numbers; edit distance and log gap of the closest **unmatched** pair | See §2 |
| Combined (1) | 0.4·name + 0.6·address token-set ratio when both addresses exist, else the name score | Lets one split use both |
| Blocking context (9) | blocking name/address cosines, rank, reverse rank; the five blocker flag bits | How the pair was found and how it ranked |
| Candidate context (4) | `n_cands` per S1; `n_s1_for_id` (S1s competing for the record); candidates of the record with blocking name cosine ≥ 0.9; gap to the S1's best blocking name similarity | Competition between S1s for one record |
| Source (2) | `is_s3`; native-script flag of the S2/S3 record | S2 and S3 have different noise |

The per-record context (`n_s1_for_id`, near-identical-name count) is counted over the **unpruned**
candidate file, which covers every S1 on both train and test. After `--s1-pct 20` + pruning, the
pruned train file holds only the featurised 20% of S1 while test holds all of them. Counting there
would give test ~5× the competitors seen in training, and the model is sensitive to it: inflating
these two features ×2 / ×3 / ×5 on the dev valid fold drops macro F0.5 0.9848 → 0.9831 / 0.9803 /
0.9782, all through recall. TF-IDF vectorisers are fitted once
(`char_wb`, 50k vocabulary, random 200k-text train subsample), saved as `models/tfidf_*.joblib`,
and reused for every dataset. **Country is never a feature.**

## 2. Address numbers: typo vs neighbour

On the dev sample, the model's errors split sharply by how the numbers in the two addresses compare:

| Numbers in the two addresses | False matches | Correct matches | Missed matches |
|---|---|---|---|
| same numbers | 12.8% | 65.6% | 16.1% |
| partial overlap | 23.7% | 18.4% | 24.7% |
| no overlap | 16.9% | 3.6% | 11.4% |
| no overlap, 1 digit apart | 13.6% | 0.9% | 5.3% |
| numbers on one side only | 31.9% | 8.6% | 40.4% |
| none on either side | 1.2% | 3.0% | 2.2% |

A one-digit difference was usually a **neighbouring building**, not a typo:
`2579` vs `2582 roanoke avenue`, `15536` vs `15537 ranford blvd`. Real typos look different:
`748` vs `548`, `22502` vs `2502`. So the features compare the closest pair of numbers that did
**not** match, and give the model both measures:

| Pair | Edit distance | Log gap | Reading |
|---|---|---|---|
| 2579 / 2582 | 2 | 1.4 | neighbour → against |
| 748 25 / 548 25 | 1 | 5.3 | typo (25 shared) → for |
| 22502 / 2502 | 1 | 7.8 | dropped digit → for |
| 60 36 / 60 39 | 1 | 1.4 | neighbouring flat → against |

Effect on the sample valid fold: macro F0.5 0.9808 → **0.9851**, entities with a false match 558 → **436**.

## 2b. First-token and relative-to-best features (PR #17, in the final submission)

The final model's wrong pairs are mostly **orphans** (78%): records that belong to no S1, usually a
copy of the true record with the **first word swapped** (`Kline Platinum Land Care` vs `Hurley Platinum
Land Care Corp`, same address), or a sibling at the same address. Test has ~40% orphan records vs 26%
in train, so this error grows on test.

Seven features, all per pair or differences, **never counts**:

| Feature | What |
|---|---|
| `first_tok_eq` | first name word equal (1) / different (−1) / missing (0) |
| `first_tok_jw` | Jaro-Winkler of the first words |
| `rest_tsr` | token-set ratio of the name after the first word |
| `first_swap` | first word differs **and** the rest is identical (≥ 0.95) |
| `name_tsr_rel`, `addr_cos_rel` | similarity minus the best *other* candidate of the same S1 (vs 0 for a single candidate) |
| `flag_key` | the pair came from an exact-key route (flag bit 32) |

Full-train valid fold (20% S1, pruned candidates **before** key routes, same folds):

| | 39 | + first-token (43) | + first-token + relative (45) |
|---|---|---|---|
| macro F0.5 | 0.9601 | 0.9629 | **0.9630** |
| orphan-weighted (test orphan rate) | 0.9578 | 0.9613 | **0.9617** |
| orphan / other false matches | 2,018 / 553 | 1,485 / 749 | **1,191 / 612** |
| matches per S1 | 3.19 | 3.20 | 3.19 |

(Scored on S1 with candidates only; `predict.py`'s full evaluation gives 0.9582 → 0.9615.)

**Pre-upload test check.** Test features for the 10.7M pruned test pairs, scored by both models:

| Test matches per S1 | France | India | US | Total matches |
|---|---|---|---|---|
| production model (39) on these features | 3.166 | 3.174 | 3.289 | 5,573,341 |
| production model on AWS (reference) | 3.17 | 3.17 | 3.29 | |
| 45 features | 3.172 | 3.185 | 3.282 | 5,578,982 (+0.1%) |

The first row reproduces the production numbers, so the check itself is sound. The new model moves
test volume by 0.1% (the rejected peer features: +7.6%). New-feature means match between train and test
(`name_tsr_rel` −0.095 / −0.102, `addr_cos_rel` −0.171 / −0.169, `first_swap` 0.049 / 0.049).

**With key routes (AWS):** valid 0.9635 → **0.9669**, false-match entities 2,882 → 2,226, singleton F0.5
0.9622 → 0.9716, **leaderboard 0.9563 → 0.9598**. Test matches +0.15% vs train-valid +0.18%.

**Upload gate** (calibrated on three uploads): the change in test matches must track the change on the train
valid fold. Key routes: +1.6% / +1.6% (LB +0.006). These features: +0.15% / +0.18% (LB +0.0034). Peer features:
+7.6% / +0.6% (LB −0.03). A fixed per-country band is too strict: key routes moved France −2.8% and still gained.

## 3. Model

LightGBM binary classifier: 127 leaves, learning rate 0.05, feature/bagging fraction 0.8,
L1 0.1 / L2 1.0, `scale_pos_weight = neg/pos`, up to 6000 rounds with early stopping (80).

**Folds**, all a deterministic function of the numeric S1 id:
- valid fold: `(id // 100) % 5 == 0` (20%). Used only for threshold tuning and reporting.
- early-stopping fold: `(id // 100) % 10 == 1` (10% of the train fold).
- fit: the rest of the train fold.

The first version early-stopped on the valid fold and tuned on it too, which made its reported
numbers optimistic. All numbers on this page come from the separated folds.

## 4. Decision

1. Keep pairs with probability ≥ `t`.
2. **One-owner assignment:** each S2/S3 record goes only to its highest-probability S1 (every record
   belongs to at most one S1 in the ground truth).
3. `t` maximises macro F0.5 on the valid fold, singletons included, with one-owner applied **inside**
   the search. Tuning without it picks a threshold for a rule that is never submitted.

Grid: 0.01 steps to 0.90, then 0.001 steps to 0.999 (the optimum sits at 0.95–0.97). A precision floor
(micro precision ≥ 0.987) was used for the first full-train model; it chose `t` = 0.97 over the
unconstrained 0.96 and cost 0.0011, so it is off by default (`--min-precision` to enable).

## 5. Validation

| | Dev sample (5% S1) | Full train, first model | Full train, skew-fixed | Full train, + key routes | + key routes + 46 features |
|---|---|---|---|---|---|
| macro F0.5 | 0.9851 | 0.9415 | 0.9580 | 0.9635 | **0.9669** |
| micro precision / recall | 0.994 / 0.969 | 0.989 / 0.874 | 0.990 / 0.911 | 0.990 / 0.925 | 0.992 / 0.928 |
| public leaderboard | — | 0.935 | 0.950 | 0.9563 | **0.9598** |
| distractors per S1 | ~12× fewer | as in test | as in test | as in test | as in test |

The sample over-states every score: each S1 competes with far fewer look-alike records. The full-train
valid fold predicted the leaderboard within 0.007–0.008 for every model. Run-to-run noise on the
full-train valid fold is about ±0.0004, so smaller differences are not evidence. Use the sample for fast
relative comparisons only, and check test-side prediction volume before uploading (§6).

**Unseen country (France proxy).** Trained on one country, scored on the other (dev sample),
threshold tuned on the training country:

| Held-out | Seen | Unseen | Drop | Threshold loss vs oracle |
|---|---|---|---|---|
| India (trained on US) | 0.9824 | 0.9707 | −0.012 | 0.0001 |
| US (trained on India) | 0.9864 | 0.9803 | −0.006 | 0.0002 |

The drop is mostly recall; a single global threshold transfers.

## 6. Tried and not kept

| Idea | Result (dev sample valid) | Decision |
|---|---|---|
| Second-stage re-ranker on out-of-fold stage-1 probabilities + rank/margin within record and S1 | 0.9850 vs 0.9850, 74 more entities with a false match | dropped; the stage-1 context features already carry it |
| `learning_rate` 0.1 | 0.9846 vs 0.9849 | kept 0.05 |
| no `scale_pos_weight` | 0.9845–0.9847; best threshold moves to ~0.71 | kept weighting |
| more rounds on the sample (3000 cap vs 1000) | same F0.5, false-match entities 724 → 558 | kept, and raised to 6000 for full train |
| **Peer features** (9): agreement with the S1's other candidates, within-S1 rank, same-name-different-number count, DBA alias | full-train valid 0.9582 → 0.9612, false-match entities −24%; **public leaderboard 0.92 vs 0.950** | **rejected** (see below) |
| Orphan negatives weighted ×1.8 (training + threshold), to match the test orphan rate | full train, orphan-weighted valid: 0.9577 → 0.9580 | not worth a rerun; the orphan shift explains only ~0.0024 of the valid→LB gap |
| 63 / 255 leaves, `min_child_samples` 50, separate threshold for empty-address pairs | full train: all within ±0.0002 of 0.9611 | plateau; kept 127 leaves, one threshold |
| "Twin": another candidate of the S1 has the same house number and a better name | full train: 0.9613 → 0.9623, false-match entities −12% | **rejected by the gate**: train-valid matches +0.14% vs test +2.20%; fires more on test (18.0% vs 16.5%) because test has more records per S1 |

**Why the peer features failed on test.** They targeted the dominant false positive (a sister branch one
number away, attached to an S1 that already has its true matches) and separated classes well on train.
But test has 5.8 S2/S3 records per S1 against 4.7 in train, and more pruned candidates per S1 (6.2 vs
5.1). The raw-count features (`peer_n_agree`, `s1_same_name_diff_num`) therefore took values beyond the
training range, and test predictions grew 7.6% (train +0.6%), almost all false merges.

The valid fold could not show this: it is drawn from the train distribution. The check that would have
caught it is cheap and should precede any upload: **compare predictions per S1 on test with train.**
Features that count other candidates must be shares or ranks, not raw counts, when candidate density can
differ between splits.

## 7. Remaining errors

- **One address empty or without numbers:** 32% of false matches and 40% of missed matches on the
  sample. The name is the only evidence, and common names cannot be separated.
- **Chains and sister branches:** identical name, different building. The number features target these.
- **Never a candidate:** ~20% of missed matches on the sample, 43% on full train. Only retrieval can fix these.
