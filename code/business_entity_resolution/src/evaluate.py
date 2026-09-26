"""Macro F0.5 scorer (the competition metric) and blocking diagnostics.

Per S1 entity: F0.5 = 1.25*tp / (n_pred + 0.25*n_true), which equals
(1.25*P*R) / (0.25*P + R); an entity with no true and no predicted matches
scores 1.0. The macro score averages over every S1 entity being evaluated.

CLI (scores files against the training ground truth; only S1 ids present in the
file are evaluated):
    python -m src.evaluate --matching path/to/matching_results.tsv
    python -m src.evaluate --candidates path/to/candidate_pairs.tsv
"""
import polars as pl

from . import io_utils as io


def per_entity_f05(pred_pairs: pl.DataFrame, gt_s1: pl.DataFrame,
                   gt_pairs: pl.DataFrame) -> pl.DataFrame:
    """Per-entity scores for every S1 id in `gt_s1` (s1_id, n).

    pred_pairs / gt_pairs: (s1_id, id). Predictions for S1 ids outside gt_s1 are ignored.
    Returns s1_id, n_true, n_pred, tp, f05.
    """
    ids = gt_s1.select("s1_id", pl.col("n").alias("n_true"))
    pred = pred_pairs.select(pl.col("s1_id", "id").cast(pl.String)).unique().join(ids.select("s1_id"), on="s1_id", how="semi")
    n_pred = pred.group_by("s1_id").len("n_pred")
    tp = pred.join(gt_pairs.select("s1_id", "id"), on=["s1_id", "id"], how="semi").group_by("s1_id").len("tp")
    return (
        ids.join(n_pred, on="s1_id", how="left").join(tp, on="s1_id", how="left")
        .with_columns(pl.col("n_pred", "tp").fill_null(0).cast(pl.UInt32))
        .with_columns(
            pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
            .otherwise(1.25 * pl.col("tp") / (pl.col("n_pred") + 0.25 * pl.col("n_true")))
            .alias("f05")
        )
    )


def macro_f05(pred_pairs, gt_s1, gt_pairs) -> float:
    return per_entity_f05(pred_pairs, gt_s1, gt_pairs)["f05"].mean()


def score_report(pred_pairs, gt_s1, gt_pairs, s1: pl.DataFrame | None = None) -> dict:
    """Macro F0.5 plus micro precision/recall and a singleton / country breakdown.

    `s1` (entity_id, country) is optional; when given, adds per-country macro F0.5.
    """
    e = per_entity_f05(pred_pairs, gt_s1, gt_pairs)
    tp, n_pred, n_true = e["tp"].sum(), e["n_pred"].sum(), e["n_true"].sum()
    single = e.filter(pl.col("n_true") == 0)
    rep = {
        "entities": e.height,
        "macro_f05": e["f05"].mean(),
        "micro_precision": tp / n_pred if n_pred else float("nan"),
        "micro_recall": tp / n_true if n_true else float("nan"),
        "singleton_share": single.height / e.height,
        "singleton_f05": single["f05"].mean() if single.height else float("nan"),
        "non_singleton_f05": e.filter(pl.col("n_true") > 0)["f05"].mean(),
        "entities_with_false_pos": e.filter(pl.col("n_pred") > pl.col("tp")).height,
    }
    if s1 is not None:
        by = (e.join(s1.select(pl.col("entity_id").alias("s1_id"), "country"), on="s1_id", how="left")
              .group_by("country").agg(pl.col("f05").mean()).sort("country"))
        rep["f05_by_country"] = dict(by.iter_rows())
    return rep


def blocking_report(cand_pairs, gt_s1, gt_pairs) -> dict:
    """Recall ceiling of a candidate set: how many true matches survived blocking,
    and the macro F0.5 of an oracle that picks exactly the true matches among the candidates."""
    cand = cand_pairs.select(pl.col("s1_id", "id").cast(pl.String)).unique().join(gt_s1.select("s1_id"), on="s1_id", how="semi")
    hit = cand.join(gt_pairs.select("s1_id", "id"), on=["s1_id", "id"], how="semi")
    per = (gt_s1.select("s1_id", "n")
           .join(hit.group_by("s1_id").len("hit"), on="s1_id", how="left")
           .with_columns(pl.col("hit").fill_null(0)))
    has_true = per.filter(pl.col("n") > 0)
    return {
        "entities": gt_s1.height,
        "candidates_total": cand.height,
        "candidates_per_s1": cand.height / gt_s1.height,
        "pair_recall": hit.height / gt_pairs.join(gt_s1.select("s1_id"), on="s1_id", how="semi").height,
        "entities_full_recall": (has_true["hit"] == has_true["n"]).mean(),
        "entities_zero_recall": (has_true["hit"] == 0).mean(),
        "oracle_macro_f05": macro_f05(hit, gt_s1, gt_pairs),
    }


def print_report(rep: dict, title: str = "") -> None:
    if title:
        print(f"== {title}")
    for k, v in rep.items():
        if isinstance(v, dict):
            print(f"  {k}: " + ", ".join(f"{a}={b:.4f}" for a, b in v.items()))
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v:,}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Score submission files against the training ground truth.")
    ap.add_argument("--matching", help="matching_results.tsv to score")
    ap.add_argument("--candidates", help="candidate_pairs.tsv to analyse")
    args = ap.parse_args()
    gt_s1_all, gt_pairs_all = io.load_gt()
    s1_train = io.load_source("train", 1, ["entity_id", "country"])

    def restrict(sub_s1):
        missing = sub_s1.join(gt_s1_all, on="s1_id", how="anti").height
        if missing:
            raise SystemExit(f"{missing} S1 ids in the file are not in the training ground truth")
        return gt_s1_all.join(sub_s1.select("s1_id"), on="s1_id", how="semi")

    if args.matching:
        sub_s1, pairs = io.read_id_list_file(args.matching, io.MATCH_COL)
        print_report(score_report(pairs, restrict(sub_s1), gt_pairs_all, s1_train), args.matching)
    if args.candidates:
        sub_s1, pairs = io.read_id_list_file(args.candidates, io.CAND_COL)
        print_report(blocking_report(pairs, restrict(sub_s1), gt_pairs_all), args.candidates)
