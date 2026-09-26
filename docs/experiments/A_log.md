# Track A experiment log

Valid fold = S1 entities with `fold == "valid"` (see `contracts.load_s1_folds`). One row per experiment; newest at the bottom.

| date | commit | dataset | change | metrics |
|---|---|---|---|---|
| 2026-09-26 | phase-0 | sample_5pct | stub C2 (name-prefix-4 + postcode keys, cap 50) | pair_recall 0.428, cands/S1 15.8, oracle F0.5 0.523 |
| 2026-09-26 | A_01 EDA | sample_5pct | EDA: no usable postcodes; 14% of matches share no name token; Indic only in S2/S3 India | see docs/method/blocking.md |
| 2026-09-26 | normalize v1 | sample_5pct | normalize v1 vs stub on 60k GT pairs | name-token-shared 0.859→0.914 (native-script 0.523→0.943); addr tset p10 74→83 |
| 2026-09-26 | normalize v2 | train | learned per-country place map (94 variants) | India S2/S3 records with canonical state name: 0.97 |
| 2026-09-26 | blocking v1 | sample_5pct | name/addr/combo/place/char4/reverse, MAX_DF=3000, cap 60 | recall 0.992, 51.5 cands/S1, oracle 0.998 |
| 2026-09-26 | blocking v1 | train 20k-S1 slices | full-density check | place-combo recall India 0.88 / US 0.89; global name/addr 0.54-0.66 |
| 2026-09-26 | blocking v1 | train (full) | full-train run, valid fold | recall 0.945, 49.1 cands/S1, full-recall 0.850, oracle 0.979; 70.5 min (India 57, US 11) |
