# Track A experiment log

Valid fold = S1 entities with `fold == "valid"` (see `contracts.load_s1_folds`). One row per experiment; newest at the bottom.

| date | commit | dataset | change | metrics |
|---|---|---|---|---|
| 2026-09-26 | phase-0 | sample_5pct | stub C2 (name-prefix-4 + postcode keys, cap 50) | pair_recall 0.428, cands/S1 15.8, oracle F0.5 0.523 |
