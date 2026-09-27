# Track B experiment log

Valid fold = S1 entities with `fold == "valid"` (see `contracts.load_s1_folds`). One row per experiment; newest at the bottom.

| date | commit | dataset | change | metrics |
|---|---|---|---|---|
| 2026-09-26 | 8b45383 | sample_5pct | first matcher: 20 features, LightGBM, one-owner (valid fold also used for early stopping — optimistic) | macro F0.5 0.9754, P 0.9874, R 0.9581, FP entities 898 |
| 2026-09-26 | ea2481d | sample_5pct | review #8 fixes (separate ES fold, one-owner inside threshold search), 33 features incl. name frequency, chunked features | 0.9808, P 0.9901, R 0.9672, FP entities 724 |
| 2026-09-26 | ea2481d | sample_5pct | round cap 1000 → 3000 (early-stopped at 1974) | 0.9808, P 0.9923, R 0.9621, FP entities 558 |
| 2026-09-26 | — | sample_5pct | second-stage re-ranker (OOF stage-1 probs + rank/margin context) | 0.9850 vs stage 1 0.9850, FP entities 520 vs 446 — dropped |
| 2026-09-26 | d6f768e | sample_5pct | address-number features (39 features) | **0.9851**, P 0.9940, R 0.9690, FP entities 436, singleton F0.5 0.9827 |
| 2026-09-27 | — | sample_5pct | leave-one-country-out (France proxy) | unseen India 0.9707 (−0.012), unseen US 0.9803 (−0.006); transferred threshold within 0.0002 of oracle |
| 2026-09-27 | ec7efbc | sample_5pct | lr 0.1 and/or no scale_pos_weight, fine grid to 0.999, no guard | 0.9845–0.9847 vs baseline 0.9849 — kept lr 0.05 + weighting; cap 6000 for full train |
| 2026-09-27 | (Mayank) | train, 20% S1 | first full-train model (33 features, guard 0.987, t=0.97) | 0.9415 (US 0.9473, India 0.9329), P 0.989, R 0.874; unguarded t=0.96 → 0.9426; leaderboard 0.935 |
