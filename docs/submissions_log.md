# Leaderboard submissions (owner: Mayank)

| date | commit (mayank / yash) | valid-fold F0.5 | public LB F0.5 | notes |
|---|---|---|---|---|
| 2026-09-27 | mayank c700561 + shared/streaming-features (PR #12) / yash ea2481d (#10) | 0.9415 (full train, 20% S1, valid fold; US 0.9473, India 0.9329) | **0.935** | first submission: blocking v1 (86.8M cands, ~50/S1), LightGBM 3000 rounds on 15.1M pairs, threshold 0.97 (precision guard >= 0.987; unguarded best 0.96 -> 0.9426), one-owner; 5,374,874 matches; empty lists France 6.6% / India 7.2% / US 6.5%; validator PASS |
| 2026-09-27 | mayank 7595cce (prune) | 0.9430 (pruned, same model, t=0.96 check) | _pending_ | v2: first model + learned pruning; candidates 86.8M -> 10.7M (6.18/S1); 5,373,640 matches (v1: 5,374,874); validator PASS |
| 2026-09-27 | final-run (mayank 6758701 + yash ec7efbc) on AWS r6i.2xlarge | **0.9579** (US 0.9607, India 0.9538; P 0.9906 / R 0.9096) | **0.950747** | final: 39 features (+address-number) rebuilt on pruned pairs, LightGBM 2740 rounds on 1.57M pruned train pairs, t=0.62 + one-owner; 5,534,991 matches; candidates 10.7M (6.18/S1); validator PASS; sha256 in output/dev/final/SHA256.txt |
| 2026-09-27 | main b861aef (PR #14: skew fix 32e7cd0) on AWS | **0.9580** (US 0.9611, India 0.9533; P 0.9902 / R 0.9109) | _pending upload_ | final, skew fixed: competitor counts over unpruned C2; t=0.60; 5,572,888 matches (+37.9k vs skewed run); candidates 10.7M (6.18/S1); validator PASS |
