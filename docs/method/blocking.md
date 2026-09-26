# Track A method notes — normalisation and blocking

Owner: Mayank. Source of the numbers: `notebooks/A_01_text_eda.ipynb` (5% dev sample `sample_5pct`; test slice = entity ids with `id % 100 < 5`).

## 1. EDA findings (2026-09-26)

### 1.1 Scripts and characters
- Train S1 and all US records are pure ASCII. **Only S2/S3 India records carry Indic script.**
  - S2 India: 23% of names and 24% of addresses contain Indic script (Devanagari 13%, then Telugu, Kannada, Tamil, Gujarati, Bengali ~1.5–2% each, Malayalam, Odia, Gurmukhi < 1%). S3 India: 13% of names, 22% of addresses.
  - Names are **phonetic renderings of English words**, often mixed with Latin: `Silver एस Agro प्राइवेट लिमिटेड`, `गुड कंसल्टेंसी लिमिटेड` (= Good Consultancy Limited), `जय सॉफ्टवेयर प्रा. लि.` (= Jai Software Pvt. Ltd.).
  - In addresses the native part is **only the state name** at the end: `…, PUNE, महाराष्ट्र`, `…, TIRUPATI, ఆంధ్రప్రదేశ్`, `…, ಕರ್ನಾಟಕ`.
  - unidecode renders them badly but consistently: `limittedd`, `limittett`, `elelpii` (LLP), `li` (लि. = Ltd), `praaivett` (private), `knslttensii` (consultancy). These are among the top-10 last tokens of S2/S3 India names.
- Other non-ASCII in S2/S3 (US and India): **injected acute accents** on vowels (`í é ó á` — `Prívate`, `Dívita`, `Léarning`), zero-width non-joiners (U+200C) inside Indic text, a few cp1252 mojibake control chars (`\x80 \x93 \x99`).
- Test France: 16% of S1 names and 28% of S1 addresses are non-ASCII — real French orthography (é, è, ç, ô, ë) — S2/S3 France ~24%.

### 1.2 Name noise (S2/S3; S1 is clean)
| pattern | share of S2/S3 names | example (S1 → S2/S3) |
|---|---|---|
| leading junk `# @ >> -- *** ... ( [` | 2.3% | `Gujarat Systems Private Limited` → `>> Gujarat Systems Limited [Center]` |
| token wrapped in `[ ]` (usually last) | ~2.6% | `Infra Rajalakshmi Gallery Corporation` → `Infra Rajalakshmi Gallery [Corp]` |
| token wrapped in `( )` | ~5% | `Meerut Finance Private Limited` → `Meerut Finance Private (Limited)` |
| **name squashed into a domain** | 4.5% | `Downtown Auto Body` → `downtownautobody.com`; `Suraj Trading Partners` → `M/s SURAJTRADINGPARTNERS.COM` |
| **random alias + DBA** (S3 only) | 1.2% | `Harbor Esports L.L.C.` → `Lyraxylo Co DBA Harbor Esports L.L.C.` |
| `aka` (S3) | 0.2% | |
| honorific prefix `m/s`, `sri`, `shri`, `shrii`, `smt`, `mr`, `dr`, `the` | India ~1–1.5% each; `the` ~1% | `Sunrise Projects Private Limited` → `Smt Sunrise Projects (Private)` |
| ALL CAPS | S2 29%, S3 8% | |
| character typos, digit-for-letter | frequent | `Precision Realty` → `Precisi0n Rlarlty`; `Ridge Project IV` → `ridgepr0jectiv.com` |
| id / tag suffix | rare | `… Pvt. Ltd. [Services] (ID: 85929)` |
| duplicated legal words | rare | `… private limited limited` |
| **name replaced by a random made-up word** | a few % | `Atlantic Center PLLC` → `Zetazeph`; `Rachel Solutions Private Limited` → `Irizeph` (address identical) |

### 1.3 Legal forms (first/last tokens)
- US S1 last tokens: `llc` (27% of US S1), `inc`, `p c`/`pc`, `pllc`, `corp`, `lp`, `co`, `corporation`, `ltd`. S2/S3 additionally move them to the **front** (`llc …`, `inc …` are top-5 first tokens).
- India S1 last: `limited` (59%), `ltd`, `llp`, `co`, `company`, `corporation`, `trust`, `society`. S2/S3 front: `private`, `limited`, `pvt`.
- **France (test)**: last tokens `sarl` (28% of France S1), `sas` (20%), `eurl`, `sa`, `sasu`, `sci`, `ei`, `cie`; first tokens `maison`, **`ets` / `etablissements`** (an abbreviation pair inside the test data), `centre`, `ecole`, `association`, `amicale`, `federation`, `comite`, `club`, `union`, `saint`, `clinique`, `lycee`, `pharmacie`, `ehpad`.

### 1.4 Addresses
- **Postcodes are essentially absent.** India 6-digit PIN in < 1% of addresses; the 5-digit numbers in ~10% of US addresses are **house numbers** (`22727 MIRAMAR BEND DR`), not ZIPs; France 5-digit < 1%. → the `postcode` column of C1 will be mostly empty; the stub's "postcode" key was in fact a house-number key.
- Empty address: S1 never; S2/S3 3.3–4.1%. Null placeholders inside addresses: `null`, `NULL`, `<NULL>`, `N/A`.
- **Component order is shuffled** (comma-separated parts): `Brandenburg, 510 River Rd, # LOT 15, Kentucky`, `MN, Isabella, 1298 Wander Road`.
- **State level differs between sources**: US S1/S2 use codes (`IN`, `NY`), S3 uses full names (`Indiana`, `Virginia`) and appends `CITY` (`Mansfield CITY`). India: English state names, codes (`TN`, `MH`) or native script. France: S1 uses the **region** (`Pays de la Loire`, `Nouvelle-Aquitaine`), S2/S3 often the **department** (`Loire-Atlantique`, `Gironde`, `Nord`) → region/department must be a weak signal, never a hard key.
- Street abbreviations: US `ST RD AVE BLVD DR LN CT CIR CV`; India `H.NO`, `NO.`, `FLR`, `BLD`, `C/O`, `S/O`, `W/O`, landmarks `near/opp/behind` (India S1 13%, S2 11%); France `R.` (rue), `AVE`, `BD`, `Imp.` (impasse), `St.-herblain` (Saint-), `bis`/`ter` after numbers.
- Address typos too: `HAMLT ST`, `KET, OH` (Kent), `AVE DES SPORST`.

### 1.5 What true matches share (true matches vs random same-country pairs, stub-normalised text)
| evidence | true matches | random pairs |
|---|---|---|
| name shares a token (> 2 chars) | 0.86 | 0.18 |
| name first token equal | 0.74 | 0.001 |
| address shares a token | 0.95 | 0.14 |
| first number in address equal (both have one) | 0.79 | 0.005 |
| median name token_set_ratio | 100 | 34 |
| median address token_set_ratio | 94 | 37 |

- **14% of true matches share no name token** (India 66% of these). Half of them are native-script names; the rest are domain-squashed names, made-up replacement names, and heavy typos. For all of them the **address** (or character n-grams of the space-less name) is the only route.
- Multiplicity: 85% of non-singleton S1s have matches in **both** S2 and S3; typically 1–3 per source (max 5 S2, 6 S3).

## 2. Design decisions that follow

### normalize.py v1 — implemented 2026-09-26
- Indic dictionary learned from train GT pairs (names + addresses): 1,236 tokens, 66 s; covers ~91% of Indic token occurrences even when learned from the 5% sample only. Stored in `cache/norm/indic_dict.parquet`.
- Full run (train + test + sample, 23M records, one process per split): ~4 min, peak ~5 GB per process.
- Effect on true-match pairs (sample, 60k GT pairs), stub normalisation → normalize v1:

| | stub | normalize v1 |
|---|---|---|
| name shares a token (all pairs) | 0.859 | 0.914 |
| name shares a token (native-script pairs) | 0.523 | **0.943** |
| name `ratio` p10 / p25 / p50 | 64 / 75 / 88 | 67 / 84 / 100 |
| address `token_set_ratio` p10 / p25 / p50 | 74 / 86 / 94 | 83 / 94 / 100 |

### normalize.py v2 — learned place map (2026-09-26)
- `learn_place_map`: per country, a digit-free S2/S3 address component that S1 of the same country almost never uses (S1 frequency < 5% of S2/S3 frequency) is linked to the S1 component it co-occurs with in ≥ 80% of its matched pairs (1.5M train GT pairs, 10 s). 94 variants: all Indian state codes (`mh dl up ka tn wb gj tg hr rj kl br mp ap pb od`), transliteration leftovers (`tamil`, `pshcimbng`→west bengal, `keralam`, `andhra`), city aliases (`bombay`→mumbai, `poona`→pune), US (`ft worth`, `street louis`→saint louis, suburbs → city/state).
- The target is **appended** (`mh` → `mh maharashtra`), never replacing, and only for records of the country it was learned in (`tn` = Tamil Nadu only in India; 0 of 138k US `tn` records touched). France has no labels → no map.
- India records with a canonical state name: S2 97.1%, S3 96.9% (S1 100%).
- Full run now ~8 min (per-country mapping adds a pass).

Open items: abbreviation mining from GT pairs; a phonetic key for typos.


### normalize.py
1. NFKC; drop zero-width chars and C1 control chars; fold accents (é→e) — needed for injected accents and for French.
2. **Indic → Latin with a learned dictionary**, not raw unidecode: build `token → english` from train GT pairs where the S2/S3 side is native script and S1 is Latin (align by position / best char similarity of the unidecode form). Fallback: unidecode + phonetic simplification (collapse doubled vowels/consonants, `tt→t`, `dd→d`, drop trailing `a`/`i` doubling). Also map native state names → English via the same pair mining on addresses (only ~30 values).
3. Strip leading junk and bracket characters (keep the bracketed content), `(ID: …)` suffixes, honorifics (`m/s`, `sri`, `shri`, `shrii`, `smt`, `mr`, `dr`, `the`).
4. **DBA / aka**: keep the part after `dba`/`aka` as the name (store the alias separately as an extra column).
5. **Domain names**: strip `www.` and the TLD, mark `is_domain`; downstream compares the space-less form.
6. Legal forms anywhere in the name (front, back, in brackets) → canonical `legal_form`, removed from `name_core`. Include French forms (`sarl sas sasu eurl sa sci snc ei cie`) — spotted in the test data itself.
7. Extra C1 columns (allowed by the contract): `name_nospace` (name_core without spaces, for domain-squashed names), `name_alias` (text before DBA/aka), `is_domain`, `addr_street` (the address component holding the house number), `addr_places` (list of address components without digits — city, district, state/region, in any order). A learned city/region split was tried and dropped: no frequency or co-occurrence rule separated cities from regions for all three countries (France S1 often ends with the city; `mumbai`/`ground floor` behave like regions in India) — compare `addr_places` as a set instead.
8. `postcode` stays in C1 but will be nearly always empty; `house_no` = first number-like token (incl. `12/3`, `59A`, `8-2-67/1`), `bis/ter` dropped.
9. Null placeholders (`null`, `<null>`, `n/a`, `none`) removed from addresses.

### blocking.py (revised blocker set; postcode key dropped)
1. **Name char n-gram TF-IDF on `name_nospace`** (char 3-grams, no word boundaries) — handles typos, domain-squashing, token splits/merges.
2. **Address TF-IDF** on `addr_clean` (word + char n-grams) — the only route for made-up names and most native-script names.
3. **Name + address TF-IDF** — disambiguates generic names.
4. **Exact keys**: `(house_no, first street token)`, `(name_core)`; skip keys shared by > 200 records.
5. **Reverse direction**: each S2/S3 record → top-3 S1s (records have exactly one owner).
Per-source top-K so S2 and S3 each get slots (85% of S1s match both).

## 3. blocking.py v1 (2026-09-26)

Per country; S2 and S3 form one pool; sparse TF-IDF top-K with `sparse_dot_topn`; absolute document-frequency cap `MAX_DF = 3000` (a relative cap made postings — and cost — grow with data size: 0.005 on 4M records = 20k postings).

| blocker (flag) | vectors | K |
|---|---|---|
| name (1) | `name_core` word 1–2-grams | 10 per source |
| address (2) | `addr_clean` words | 10 per source |
| combined (16) | name ⊕ address (cosine = mean) | 10 per source |
| place-conditioned combined (16) | same, searched within every shared `addr_places` value (groups ≤ 300k) | 10 |
| char4 (4) | `name_nospace` char 4-grams, only against single-token/domain names | 5 |
| reverse (8) | each S2/S3 record → top-3 S1 on the combined vector | 3 |

Pairs are scored (`blk_name_sim`, `blk_addr_sim` = cosines), ranked, capped at 60 per S1 (a record's best S1 is kept up to rank 120).

### Results
**Sample (`sample_5pct`, valid fold)**: pair recall **0.992**, 51.5 candidates/S1, entities with full recall 0.973, oracle F0.5 **0.998**, 117 s.

**Full density (20k-S1 slice vs the full train pool of the country)** — the sample is optimistic (12× fewer distractors):

| recall | India (pool 4.1M) | US (pool 6.2M) |
|---|---|---|
| name alone | 0.54 | 0.63 |
| address alone | 0.66 | 0.59 |
| place-conditioned combined | **0.88** | **0.89** |
| all (reverse inflated by the slice) | 0.90 | 0.91 |

Generic names ("… private limited") and the df cap (which removes city/state words) make the global name/address searches weak at full density; conditioning on a shared address place is what separates them. Slice runs take 270–400 s, dominated by fitting/transforming the full pool.

### Full train (2026-09-26) — the real numbers
Valid fold of full train (all 2.2M S1 blocked against the full pools): **pair recall 0.945**, **49.1 candidates/S1**, entities with full recall 0.850, entities with zero recall 0.008, **oracle F0.5 0.979**. 108.3M pairs, 956 MB Parquet.
Time 70.5 min on the 16 GB laptop: India 883k x 4.1M in 57 min (large state-level place groups make the place-conditioned search expensive), US 1.32M x 6.2M in 11 min. Peak: ~8 GB working set, commit charge 33 of 34 GB and free RAM down to 0.3 GB while validating/writing C2 (`contracts.write_candidates` loads every column of all records — make it load only ids + country).

| blocker | recall alone | true pairs found only by it | pairs |
|---|---|---|---|
| name words | 0.598 | 5,250 | 41.6M |
| address words | 0.599 | 1,148 | 39.8M |
| combined + place-conditioned | **0.917** | **53,994** | 60.6M |
| char 4-grams (squashed names) | 0.051 | 14,599 | 9.8M |
| reverse | 0.867 | 5,500 | 22.8M |

### Test (2026-09-26)
86.8M pairs, 52 min at below-normal priority (France 1 min, India 45 min, US 4.5 min); peak free RAM stayed >= 1.4 GB with the lighter C2 check (#9).

| country | S1 | candidates/S1 (mean / median / max) | S1 without candidates |
|---|---|---|---|
| France | 259,452 | 54.0 / 57 / 109 | 23 |
| India | 809,986 | 49.6 / 51 / 116 | 421 |
| US | 663,106 | 49.2 / 50 / 108 | 60 |

France (no labels) gets the same candidate volume as the labelled countries (train median 50).

### Next
1. Recall 0.945 → ≥ 0.95: raise `K_PLACE`, place search per source, a street-level key (`house_no` + street word) within place groups.
2. Recall: raise `K_PLACE`, search place groups by source, add a street-level key (`house_no` + street word) within place groups.
3. Speed: fit vectorizers once per country and reuse for train/test; cache matrices.
