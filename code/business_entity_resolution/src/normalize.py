"""Text normalisation: raw records -> C1 (see contracts.py and docs/method/blocking.md).

    python -m src.normalize                    # learn Indic dictionary, normalise train + test + sample
    python -m src.normalize --splits sample_5pct   # dev loop: only the sample

Everything is vectorised Polars except Indic transliteration, which runs once per
unique string containing Indic characters.
"""
import re
import time
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz.distance import JaroWinkler
from unidecode import unidecode

from . import config
from . import contracts as C

PRODUCER = "normalize v1"

# ---------------------------------------------------------------- vocabularies

HONORIFICS = ["m s", "ms", "messrs", "sri", "shri", "shree", "shrii", "smt", "mr", "mrs", "dr", "the"]

# legal token -> canonical part; canonical legal_form = sorted unique parts joined by "_"
LEGAL = {
    "inc": "inc", "incorporated": "inc", "llc": "llc", "pllc": "pllc", "llp": "llp", "lp": "lp",
    "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "pc": "pc", "plc": "plc", "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci", "snc": "snc",
    "ei": "ei", "cie": "co", "gmbh": "gmbh",
}
LEGAL_RE = "|".join(sorted(LEGAL, key=len, reverse=True))
# legal tokens that are never part of a business's own name: stripped anywhere, not only at the ends
LEGAL_ANYWHERE = ["inc", "incorporated", "llc", "pllc", "llp", "ltd", "pvt", "corp", "plc", "sarl", "sasu", "eurl", "gmbh"]

ABBREV_NAME = {
    "intl": "international", "int'l": "international", "mfg": "manufacturing", "svc": "services",
    "svcs": "services", "mgmt": "management", "assoc": "associates", "assocs": "associates", "bros": "brothers",
    "grp": "group", "hldgs": "holdings", "tech": "technologies", "ets": "etablissements", "ste": "societe",
    "natl": "national", "dept": "department", "univ": "university", "ctr": "center", "centre": "center",
}
ABBREV_ADDR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "bld": "building", "bldg": "building", "dr": "drive", "ln": "lane", "ct": "court",
    "cir": "circle", "cv": "cove", "pl": "place", "pkwy": "parkway", "hwy": "highway", "trl": "trail",
    "sq": "square", "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor", "flrs": "floor",
    "hno": "house", "h": "house", "no": "", "nos": "", "r": "rue", "imp": "impasse", "ch": "chemin",
    "rte": "route", "fg": "faubourg", "all": "allee", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest", "mt": "mount",
    "opp": "opposite", "nr": "near", "rly": "railway", "stn": "station", "c o": "", "s o": "", "w o": "",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "district of columbia": "dc", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "puerto rico": "pr",
}
NULL_RE = r"(?i)<\s*null\s*>|\bnull\b|\bn/a\b|\bnone\b|\bnan\b"
INDIC_RANGE = "ऀ-෿"
TLD_RE = r"\.(?:com|net|org|in|co\.in|co|io|biz|us|info|fr|online|shop|store)\b"

# ---------------------------------------------------------------- Indic transliteration

_INDIC = re.compile(f"[{INDIC_RANGE}]")
_TOKEN = re.compile(f"[{INDIC_RANGE}‌‍]+|[^{INDIC_RANGE}‌‍]+")


def _simplify(s: str) -> str:
    """Phonetic skeleton used to compare transliterations with English words."""
    s = re.sub(r"[^a-z]", "", s.lower())
    return re.sub(r"(.)\1+", r"\1", s)


def _fallback(tok: str) -> str:
    return _simplify(unidecode(tok))


def learn_indic_dict(min_votes: int = 2, min_share: float = 0.6, min_sim: float = 0.6) -> pl.DataFrame:
    """Learn native-script token -> English token from train GT pairs (names and addresses).

    For every matched pair whose S2/S3 side contains Indic script, each native token is
    aligned to the S1 token with the most similar phonetic skeleton; votes are kept
    when similarity >= min_sim. Returns (native, english, votes).
    """
    s1 = C.load_records("train", 1).select(pl.col("entity_id").alias("s1_id"), "business_name", "business_address")
    others = pl.concat([C.load_records("train", s) for s in (2, 3)]).rename({"entity_id": "id"})
    _, gt = C.load_ground_truth("train")
    j = (gt.join(others, on="id").filter(
            pl.col("business_name").str.contains(f"[{INDIC_RANGE}]") | pl.col("business_address").str.contains(f"[{INDIC_RANGE}]"))
         .join(s1, on="s1_id", suffix="_1"))
    votes: dict[str, Counter] = defaultdict(Counter)
    for n2, a2, n1, a1 in j.select("business_name", "business_address", "business_name_1", "business_address_1").iter_rows():
        for native, latin in ((n2, n1), (a2, a1)):
            if not _INDIC.search(native):
                continue
            en = [w.lower() for w in re.findall(r"[A-Za-z]+", latin)]
            en_s = [_simplify(w) for w in en]
            if not en:
                continue
            for tok in _TOKEN.findall(native):
                if not _INDIC.search(tok):
                    continue
                tok = tok.replace("‌", "").replace("‍", "")
                f = _fallback(tok)
                if not f:
                    continue
                sims = [JaroWinkler.normalized_similarity(f, e) for e in en_s]
                best = max(range(len(en)), key=sims.__getitem__)
                if sims[best] >= min_sim:
                    votes[tok][en[best]] += 1
    rows = []
    for tok, c in votes.items():
        word, n = c.most_common(1)[0]
        if n >= min_votes and n / sum(c.values()) >= min_share:
            rows.append((tok, word, n))
    return pl.DataFrame(rows, schema={"native": pl.String, "english": pl.String, "votes": pl.UInt32}, orient="row")


def indic_dict_path():
    return config.CACHE_DIR / "norm" / "indic_dict.parquet"


def load_indic_dict() -> dict[str, str]:
    d = pl.read_parquet(indic_dict_path())
    return dict(zip(d["native"].to_list(), d["english"].to_list()))


def transliterate(text: str, table: dict[str, str]) -> str:
    """Replace every Indic token by its learned English word (fallback: phonetic unidecode)."""
    out = []
    for tok in _TOKEN.findall(text):
        if _INDIC.search(tok):
            tok = tok.replace("‌", "").replace("‍", "")
            out.append(" " + table.get(tok, _fallback(tok)) + " ")
        else:
            out.append(tok)
    return "".join(out)


def _translit_expr(col: str, table: dict[str, str]) -> pl.Expr:
    return (pl.when(pl.col(col).str.contains(f"[{INDIC_RANGE}]"))
            .then(pl.col(col).map_elements(lambda s: transliterate(s, table), return_dtype=pl.String, skip_nulls=True))
            .otherwise(pl.col(col)))

# ---------------------------------------------------------------- shared text cleaning

def _ascii_lower(e: pl.Expr) -> pl.Expr:
    """Drop zero-width/control chars, fold accents, lowercase."""
    return (e.str.replace_all(r"[​-‏⁠﻿\x00-\x1f\x7f-\x9f]", "")
             .str.normalize("NFKD").str.replace_all(r"\p{Mn}", "")
             .str.to_lowercase())


def _map_tokens(e: pl.Expr, mapping: dict[str, str]) -> pl.Expr:
    """Token-wise dictionary replacement on a space-separated string."""
    return (e.str.split(" ").list.eval(pl.element().replace(mapping))
             .list.join(" ").str.replace_all(r"\s+", " ").str.strip_chars())


def _squash(e: pl.Expr) -> pl.Expr:
    return e.str.replace_all(r"\s+", " ").str.strip_chars()

# ---------------------------------------------------------------- names

def _name_exprs(table: dict[str, str]) -> list[pl.Expr]:
    return [_ascii_lower(_translit_expr("business_name", table)).alias("_n")]


def _clean_name(df: pl.DataFrame) -> pl.DataFrame:
    n = pl.col("_n")
    df = df.with_columns(
        # DBA / aka: the real name follows the marker; keep the alias for reference
        n.str.extract(r"^(.*?)\s*\b(?:dba|d/b/a|aka|a/k/a)\b\s*:?\s*(.+)$", 1).alias("name_alias"),
        pl.coalesce(n.str.extract(r"\b(?:dba|d/b/a|aka|a/k/a)\b\s*:?\s*(.+)$", 1), n).alias("_n"),
    ).with_columns(
        n.str.contains(TLD_RE).alias("is_domain"),
    ).with_columns(
        n.str.replace_all(r"\bwww\.", " ").str.replace_all(TLD_RE, " ")
         .str.replace_all(r"\(\s*id\s*:?\s*\d+\s*\)", " ")          # "(ID: 85929)"
         .str.replace_all(r"^m\s*/\s*s\b", " ")                      # "M/s"
         .str.replace_all("&", " and ").str.replace_all(r"\+", " plus ")
         .str.replace_all(r"[.'`’]", "")                              # l.l.c. -> llc, orelee's -> orelees
         .str.replace_all(r"[^a-z0-9]+", " ")
         .str.replace_all(r"([a-z])0([a-z])", "${1}o${2}")           # precisi0n -> precision
         .str.replace_all(r"([a-z])0\b", "${1}o").str.replace_all(r"\b0([a-z])", "o${1}")
         .alias("_n"),
    ).with_columns(_squash(_map_tokens(n, ABBREV_NAME)).alias("name_clean"))
    return df


def _legal_split(df: pl.DataFrame) -> pl.DataFrame:
    """Strip legal-form and honorific tokens from the ends of name_clean, and unambiguous
    legal tokens (LEGAL_ANYWHERE) from anywhere; collect them into a canonical legal_form."""
    edge = f"(?:(?:{LEGAL_RE}|{'|'.join(HONORIFICS)})\\s)+"
    tail = f"(?:\\s(?:{LEGAL_RE}))+$"
    inner = rf"\b(?:{'|'.join(LEGAL_ANYWHERE)})\b"
    df = df.with_columns(
        pl.col("name_clean").str.extract(f"^({edge})", 1).fill_null("").alias("_lead"),
        pl.col("name_clean").str.extract(f"({tail})", 1).fill_null("").alias("_tail"),
    ).with_columns(
        pl.col("name_clean").str.replace(f"^{edge}", "").str.replace(tail, "").alias("_core"),
    ).with_columns(
        pl.col("_core").str.extract_all(inner).list.join(" ").alias("_inner"),
        _squash(pl.col("_core").str.replace_all(inner, " ")).str.replace(r"^(?:and )+", "").str.replace(r"(?: and)+$", "")
          .alias("_core"),
    )
    legal_parts = (pl.concat_str("_lead", pl.lit(" "), "_tail", pl.lit(" "), "_inner").str.split(" ")
                   .list.eval(pl.element().replace_strict(LEGAL, default=None).drop_nulls().unique().sort())
                   .list.join("_"))
    return df.with_columns(
        # never return an empty core: fall back to the full clean name
        pl.when(pl.col("_core") == "").then(pl.col("name_clean")).otherwise(pl.col("_core")).alias("name_core"),
        legal_parts.alias("legal_form"),
    ).with_columns(pl.col("name_core").str.replace_all(" ", "").alias("name_nospace"))

# ---------------------------------------------------------------- addresses

def _clean_address(df: pl.DataFrame, table: dict[str, str]) -> pl.DataFrame:
    a = pl.col("_a")
    df = df.with_columns(
        _ascii_lower(_translit_expr("business_address", table).str.replace_all(NULL_RE, " ")).alias("_a"),
    ).with_columns(
        # keep commas as component separators for now
        a.str.replace_all("&", " and ").str.replace_all(r"[.'`’]", "")
         .str.replace_all(r"[^a-z0-9,/\-]+", " ").str.replace_all(r"\s*,\s*", ",")
         .str.replace_all(r",+", ",").str.strip_chars(" ,")
         .str.replace_all(r"\b0+(\d)", "${1}")                               # 0049 -> 49
         .alias("_a"),
    ).with_columns(
        a.str.split(",").list.eval(
            _map_tokens(pl.element().str.replace_all(r"[/\-]", " "), ABBREV_ADDR)
            .str.replace(r"\s+city$", "")                                  # "mansfield city" (S3)
            .replace(US_STATES)
        ).list.eval(pl.element().filter(pl.element() != "")).alias("addr_parts"),
    )
    return df.with_columns(
        pl.col("addr_parts").list.join(" ").alias("addr_clean"),
        # first number-like token of the raw (cleaned) address: 1298, 59a, 570/13, 8-2-67/1
        a.str.extract(r"\b(\d+[a-z]?(?:[/\-]\d+[a-z]?)*)\b", 1).fill_null("").alias("house_no"),
        pl.coalesce(a.str.extract(r"\b(\d{6})\b", 1), a.str.extract(r"\b(\d{3}\s\d{3})\b", 1).str.replace(" ", ""),
                    pl.lit("")).alias("postcode"),
    ).with_columns(
        # the component holding the house number, e.g. "1298 wander road"
        pl.col("addr_parts").list.eval(pl.element().filter(pl.element().str.contains(r"\d"))).list.first()
          .fill_null("").alias("addr_street"),
        # components without digits (city, district, state/region, landmarks), order-free
        pl.col("addr_parts").list.eval(pl.element().filter(~pl.element().str.contains(r"\d"))).list.unique(maintain_order=True)
          .alias("addr_places"),
    )


# ---------------------------------------------------------------- public API

EXTRA_COLUMNS = ["name_nospace", "name_alias", "is_domain", "addr_street", "addr_places"]


def _normalize(df: pl.DataFrame, table: dict[str, str]) -> pl.DataFrame:
    """All derived columns, including the internal `addr_parts` list."""
    out = df.with_columns(
        (pl.col("business_name").str.contains(f"[{INDIC_RANGE}]")
         | pl.col("business_address").str.contains(f"[{INDIC_RANGE}]")).alias("is_native_script"),
        *_name_exprs(table),
    )
    out = _clean_address(_legal_split(_clean_name(out)), table)
    return out.with_columns(pl.col("name_alias").fill_null(""))


def normalize_frame(df: pl.DataFrame, table: dict[str, str]) -> pl.DataFrame:
    """Raw records (entity_id, business_name, business_address, country) -> C1 + extra columns."""
    return _normalize(df, table).select(*C.C1_COLUMNS, *EXTRA_COLUMNS)


def normalize_all(splits=("train", "test", "sample_5pct"), relearn: bool = False) -> None:
    config.CACHE_DIR.joinpath("norm").mkdir(parents=True, exist_ok=True)
    if relearn or not indic_dict_path().exists():
        t = time.time()
        d = learn_indic_dict()
        d.write_parquet(indic_dict_path())
        print(f"indic dictionary: {d.height:,} tokens in {time.time() - t:.0f}s")
    table = load_indic_dict()

    for dataset in splits:
        t = time.time()
        for source in (1, 2, 3):
            df = normalize_frame(C.load_records(dataset, source), table)
            C.write_norm(df, dataset, source, producer=PRODUCER)
            print(f"{dataset} S{source}: {df.height:,} rows ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Normalise records into C1.")
    ap.add_argument("--splits", nargs="+", default=["train", "test", "sample_5pct"], choices=C.DATASETS)
    ap.add_argument("--relearn", action="store_true", help="re-learn the Indic dictionary")
    args = ap.parse_args()
    if len(args.splits) == 1:
        normalize_all(tuple(args.splits), args.relearn)
    else:
        # one process per split: Polars keeps freed memory, and train + test in one
        # process pushes a 16 GB machine into swapping (10x slower)
        import subprocess
        import sys
        if args.relearn or not indic_dict_path().exists():
            normalize_all((), relearn=True)
        for split in args.splits:
            subprocess.run([sys.executable, "-m", "src.normalize", "--splits", split], check=True)
