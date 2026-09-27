"""Exact-key candidate routes, added after pruning (C2 += FLAG_KEY pairs).

Why: blocking drops every token in more than MAX_DF records, so a name like "behavioral health
clinic" at "132 highway 89 mayflower ar" is searched as just "mayflower" and its identical copies
can fall outside the top K. On the full-train valid fold, 16.6k true pairs are missed by blocking;
~a third of them share an exact key with their S1:

  name   same name_core token set, and a shared address place
  addr   same addr_clean token set
  hn     same house number, and a shared address place
  name_only  same name_core token set, for records with an empty address
  street same house number + a street word (catches misspelt cities)

Key groups larger than the caps are skipped (common numbers in big cities). The pairs not
already in the wide candidates are ranked per S1 by name + address token Jaccard; the top
K_KEY with score >= SC_MIN are appended to the wide C2 (all S1, so per-id context features see
them on train and test alike) and to the pruned C2 (only the S1 it already covers). The step
is idempotent: earlier FLAG_KEY pairs are removed first. blk_name_sim / blk_addr_sim hold
the token Jaccards for these pairs.
"""
from __future__ import annotations

import os
import time

import polars as pl

from . import contracts as C
from .prune import wide_path

K_KEY = 3
SC_MIN = 0.8
CAPS = {"name": 500, "addr": 200, "hn": 300, "name_only": 200, "street": 200}
N_CHUNKS = 8
PRODUCER = "keyroute v2"
COLUMNS = ["entity_id", "country", "name_core", "addr_clean", "house_no", "addr_places"]


def _prep(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("name_core").str.split(" ").list.unique().list.sort().alias("nt"),
        pl.col("addr_clean").str.split(" ").list.unique().list.sort().alias("at"),
    )


def _keys(df: pl.DataFrame, idc: str) -> dict[str, pl.DataFrame]:
    places = (df.select(idc, "country", "name_core", "house_no", pl.col("addr_places").alias("place"))
                .explode("place", empty_as_null=True).drop_nulls("place")
                .filter(pl.col("place").str.len_chars() >= 3))
    return {
        "name": (places.filter(pl.col("name_core") != "")
                 .join(df.select(idc, pl.col("nt").list.join(" ").alias("k")), on=idc)
                 .select(idc, "country", "k", "place").unique()),
        "addr": df.filter(pl.col("addr_clean") != "").select(idc, "country", pl.col("at").list.join(" ").alias("k")),
        "hn": places.filter(pl.col("house_no") != "").select(idc, "country", "house_no", "place").unique(),
        # house number + a street word (alphabetic, >= 4 chars, not a place token): survives a misspelt city
        "street": (df.filter(pl.col("house_no") != "").select(idc, "country", "house_no", pl.col("at").alias("w"))
                   .explode("w", empty_as_null=True).drop_nulls("w")
                   .filter((pl.col("w").str.len_chars() >= 4) & ~pl.col("w").str.contains(r"\d"))
                   .join(df.select(idc, pl.col("addr_places").list.join(" ").str.split(" ").alias("w"))
                           .explode("w", empty_as_null=True).drop_nulls("w"), on=[idc, "w"], how="anti")
                   .unique()),
        # records without an address: name alone (S1 side keeps every S1 with a name)
        "name_only": (df.filter((pl.col("name_core") != "") & ((pl.col("addr_clean") == "") | (idc == "s1_id")))
                      .select(idc, "country", pl.col("nt").list.join(" ").alias("k"))),
    }


ON = {"name": ["country", "k", "place"], "addr": ["country", "k"], "hn": ["country", "house_no", "place"],
      "name_only": ["country", "k"], "street": ["country", "house_no", "w"]}


def _jac(x: str, y: str) -> pl.Expr:
    inter = pl.col(x).list.set_intersection(y).list.len()
    return (inter / pl.col(x).list.set_union(y).list.len().clip(1)).cast(pl.Float32)


def key_pairs(dataset: str, wide_c2: pl.LazyFrame) -> pl.DataFrame:
    s1 = _prep(pl.read_parquet(C.norm_path(dataset, 1), columns=COLUMNS).rename({"entity_id": "s1_id"}))
    rec = _prep(pl.concat([pl.read_parquet(C.norm_path(dataset, s), columns=COLUMNS) for s in (2, 3)])
                .rename({"entity_id": "id"}))
    krec = _keys(rec, "id")
    krec = {r: k.join(k.group_by(ON[r]).len().filter(pl.col("len") <= CAPS[r]).drop("len"), on=ON[r], how="semi")
            for r, k in krec.items()}
    rec_tok = rec.select("id", pl.col("nt").alias("bnt"), pl.col("at").alias("bat"))
    out = []
    chunk = s1.with_columns((pl.col("s1_id").hash(seed=1) % N_CHUNKS).alias("_c"))
    for c in range(N_CHUNKS):
        t = time.time()
        part = chunk.filter(pl.col("_c") == c).drop("_c")
        ks1 = _keys(part, "s1_id")
        p = pl.concat([ks1[r].join(krec[r], on=ON[r]).select("s1_id", "id") for r in ON]).unique()
        have = wide_c2.select("s1_id", "id").join(part.select("s1_id").lazy(), on="s1_id", how="semi").collect()
        p = p.join(have, on=["s1_id", "id"], how="anti")
        p = (p.join(part.select("s1_id", "nt", "at"), on="s1_id").join(rec_tok, on="id")
              .with_columns(_jac("nt", "bnt").alias("blk_name_sim"), _jac("at", "bat").alias("blk_addr_sim"))
              .drop("nt", "at", "bnt", "bat")
              .with_columns((pl.col("blk_name_sim") + pl.col("blk_addr_sim")).alias("_sc"))
              .filter(pl.col("_sc") >= SC_MIN)
              .with_columns(pl.col("_sc").rank("ordinal", descending=True).over("s1_id").alias("_r"))
              .filter(pl.col("_r") <= K_KEY))
        out.append(p)
        print(f"  key chunk {c + 1}/{N_CHUNKS}: {part.height:,} S1 -> {p.height:,} new pairs ({time.time() - t:.0f}s)", flush=True)
    p = pl.concat(out)
    return p.with_columns(
        pl.col("_r").cast(pl.UInt16).alias("blk_rank"),
        pl.col("_sc").rank("ordinal", descending=True).over("id").cast(pl.UInt16).alias("blk_rev_rank"),
        pl.lit(C.FLAG_KEY, pl.UInt8).alias("blk_flags"),
    ).select(list(C.C2_COLUMNS))


def _strip(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter((pl.col("blk_flags") & C.FLAG_KEY) == 0)


def add(dataset: str) -> pl.DataFrame:
    """Append key-route pairs to the wide and the pruned C2 of `dataset` (run after prune)."""
    cand = C.candidates_path(dataset)
    wide = wide_path(cand)
    if not wide.exists():
        raise SystemExit(f"{wide} missing: run the prune step first")
    wide_lf = pl.scan_parquet(wide).filter((pl.col("blk_flags") & C.FLAG_KEY) == 0)
    new = key_pairs(dataset, wide_lf)
    tmp = wide.with_name(wide.stem + ".tmp.parquet")
    pl.concat([wide_lf, new.lazy()]).sink_parquet(tmp, compression="zstd")
    os.replace(tmp, wide)
    pruned = _strip(pl.read_parquet(cand))
    add_p = new.join(pruned.select("s1_id").unique(), on="s1_id", how="semi")
    out = pl.concat([pruned, add_p]).sort("s1_id", "blk_rank")
    C.write_candidates(out, dataset, producer=PRODUCER)
    print(f"keyroute {dataset}: +{new.height:,} wide pairs, +{add_p.height:,} final candidates "
          f"({pruned.height:,} -> {out.height:,}, {out.height / out['s1_id'].n_unique():.2f}/S1)")
    return add_p


def report(dataset: str) -> None:
    from .evaluate import blocking_report, print_report
    gt_s1, gt = C.load_ground_truth(dataset)
    folds = C.load_s1_folds(dataset)
    cands = pl.read_parquet(C.candidates_path(dataset), columns=["s1_id", "id", "blk_flags"])
    valid = (gt_s1.join(folds.filter(pl.col("fold") == "valid").select("s1_id"), on="s1_id", how="semi")
                  .join(cands.select("s1_id").unique(), on="s1_id", how="semi"))
    print_report(blocking_report(cands, valid, gt), f"{dataset} valid fold, final candidates with key routes")
    key = cands.filter((pl.col("blk_flags") & C.FLAG_KEY) > 0).join(valid, on="s1_id", how="semi")
    true = key.join(gt, on=["s1_id", "id"], how="semi").height
    print(f"  key pairs on valid S1: {key.height:,}, true {true:,} ({true / max(key.height, 1):.3f})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Add exact-key candidate routes after pruning.")
    ap.add_argument("--dataset", default="sample_5pct", choices=C.DATASETS)
    args = ap.parse_args()
    add(args.dataset)
    if args.dataset != "test":
        report(args.dataset)
