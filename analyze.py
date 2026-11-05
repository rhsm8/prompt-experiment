"""
analyze.py  -  prompt template selection

  1. x_k = failed responses, n = total calls for template k
  2. p_k = P(X <= x_k | n, delta),  X ~ Binomial(n, delta)
  3. Benjamini-Hochberg at level alpha -> rejected set
  4. among rejected, pick the template with the smallest variability

Diagnostics (display only, NOT used for selection):
  - direction_agreement : share of the modal stance within each template
  - original vs negation flip rate by framing
  - cross-framing direction split rate
  - stance distribution by statement_version

Output: analysis/summary_<topic>_<timestamp>.txt
        (the topics are also listed at the top of the file).

Usage:
    python analysis/analyze.py results/*.csv
    python analysis/analyze.py results/*.csv --delta 0.05 --alpha 0.05
    python analysis/analyze.py results/*.csv --outdir some/other/dir
"""
import argparse
from datetime import datetime
from pathlib import Path
import glob

import numpy as np
import pandas as pd
from scipy.stats import binom


def modal_share(s):
    """Share of the most frequent stance among non-empty values (1.0 = all agree)."""
    s = s[s != ""]
    return s.value_counts(normalize=True).max() if len(s) else np.nan


def main(paths, delta, alpha, outdir):
    lines = []

    def out(text=""):
        print(text)
        lines.append(str(text))

    # ---------- load ----------
    # expand wildcards ourselves (Windows shells do not)
    files = []
    for p in paths:
        files.extend(sorted(glob.glob(str(p))) or [str(p)])
    if not files:
        raise SystemExit("No CSV files found.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # ---------- topics (first thing in the file) ----------
    topics = (
        df.groupby("topic_id")
        .agg(topic_en=("topic_en", "first"), topic_ja=("topic_ja", "first"), rows=("topic_id", "size"))
        .reset_index()
    )
    out(f"Topics: {len(topics)}")
    for _, r in topics.iterrows():
        out(f"  - {r['topic_id']} | {r['topic_en']} | {r['topic_ja']} (rows: {int(r['rows'])})")
    out("")
    out(f"Files: {len(files)}")

  # ---------- step 0: data preparation ----------
    df["failed"] = 1 - pd.to_numeric(df["parse_success"], errors="coerce").fillna(0).astype(int)
    df["canonical_score"] = pd.to_numeric(df["canonical_score"], errors="coerce")

    # neutral point of the scale: 3 for 1..5 data, 0 for -2..+2 data
    cs = df["canonical_score"].dropna()
    SCALE_MID = 0 if (len(cs) and cs.min() < 1) else 3

    df["canonical_stance"] = df["canonical_stance"].fillna("").astype(str)

    # template = statement_version x framing_id  (K = 3 x 7 = 21)
    df["template"] = df["statement_version"].astype(str) + " | " + df["framing_id"].astype(str)

    # ---------- steps 1-2: failures and p-values ----------
    t = df.groupby("template").agg(n=("failed", "size"), x=("failed", "sum")).reset_index()
    t["fail_rate"] = t["x"] / t["n"]
    t["p"] = binom.cdf(t["x"], t["n"], delta)

    # variability: variance inside each (model x language x topic) cell, averaged
    # over cells. Within-topic so that topic-to-topic differences in average stance
    # do not inflate the variability of a template.
    cell_var = (
        df.groupby(["template", "model", "language", "topic_id"])["canonical_score"].var()
        .groupby("template").mean().rename("variability").reset_index()
    )
    t = t.merge(cell_var, on="template")

    # diagnostic (not used for selection): direction agreement within each cell
    dir_agree = (
        df.groupby(["template", "model", "language", "topic_id"])["canonical_stance"]
        .apply(modal_share)
        .groupby("template").mean().rename("direction_agreement").reset_index()
    )
    t = t.merge(dir_agree, on="template")

    # ---------- step 3: Benjamini-Hochberg ----------
    t = t.sort_values("p").reset_index(drop=True)
    K = len(t)
    t["bh_threshold"] = np.arange(1, K + 1) * alpha / K
    passed = np.where(t["p"] <= t["bh_threshold"])[0]
    k_star = passed.max() + 1 if len(passed) else 0
    t["reject"] = np.arange(1, K + 1) <= k_star

    # ---------- step 4: smallest variability in the rejected set ----------
    cand = t[t["reject"]].sort_values("variability")

    fm = df.groupby(["model", "language"])["failed"].agg(["sum", "size", "mean"])
    fm.columns = ["failed", "n", "fail_rate"]

    # ---------- print: selection ----------
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)

    out(f"K = {K} templates | n per template = {sorted(int(v) for v in t['n'].unique())} "
        f"| delta = {delta} | alpha = {alpha}")
    out(f"models: {sorted(df['model'].unique())} | languages: {sorted(df['language'].unique())}")

    # best case: every template has 0 failures -> BH threshold for rank K is alpha
    n_min = int(t["n"].min())
    p_best = binom.cdf(0, n_min, delta)
    if p_best > alpha:
        out(f"\n[WARN] n={n_min} is too small: even 0 failures gives p={p_best:.3f} "
            f"> alpha={alpha}. No template can be rejected. Increase runs.")

    out("\n=== Template test (sorted by p-value) ===")
    out(t.round(4).to_string(index=False))

    out(f"\nk* = {k_star} of {K} templates pass the failure criterion")
    if len(cand):
        out("\n=== Candidate set (smallest variability first) ===")
        out(cand[["template", "x", "n", "fail_rate", "p", "variability", "direction_agreement"]]
            .round(4).to_string(index=False))
        out(f"\n>>> SELECTED: {cand.iloc[0]['template']}")
    else:
        out("\n>>> Candidate set is empty (no template rejected)")

    out("\n=== Failure rate by model x language ===")
    out(fm.round(4).to_string())

    # ---------- print: diagnostics (display only) ----------
    out("\n" + "=" * 70)
    out("DIAGNOSTICS (display only; not used for selection)")
    out("=" * 70)

    # D1) stance distribution by statement_version (parsed rows only)
    ok = df[df["canonical_stance"] != ""]
    if len(ok):
        dist = (
            ok.groupby("statement_version")["canonical_stance"]
            .value_counts(normalize=True).unstack(fill_value=0)
        )
        for col in ("support", "neutral", "oppose"):
            if col not in dist.columns:
                dist[col] = 0.0
        out("\n--- D1: stance distribution by statement_version ---")
        out(dist[["support", "neutral", "oppose"]].round(3).to_string())

    # D2) original vs negation: does canonical direction flip between them?
    #     Compared within model x language x topic x framing (mean over runs).
    mean_cs = (
        df.groupby(["model", "language", "topic_id", "framing_id", "statement_version"])
        ["canonical_score"].mean().unstack("statement_version")
    )
    if {"original", "negation"} <= set(mean_cs.columns):
        m = mean_cs.dropna(subset=["original", "negation"]).copy()
        m["flip"] = (m["original"] - SCALE_MID) * (m["negation"] - SCALE_MID) < 0
        m["gap"] = (m["original"] - m["negation"]).abs()
        flip_by_framing = m.groupby("framing_id").agg(
            cells=("flip", "size"),
            flip_rate=("flip", "mean"),
            mean_abs_gap=("gap", "mean"),
        )
        out("\n--- D2: original vs negation (canonical) by framing ---")
        out("flip_rate: share of cells where the two point in opposite directions")
        out("mean_abs_gap: mean |original - negation| on the canonical scale")
        out(flip_by_framing.round(3).to_string())
        out(f"overall flip_rate = {m['flip'].mean():.3f} "
            f"(over {len(m)} model x language x topic x framing cells)")
    else:
        out("\n--- D2 skipped: need both 'original' and 'negation' in the data ---")

    # D3) cross-framing split: within model x language x topic x version,
    #     do framings disagree on support vs oppose?
    fr = (
        df.groupby(["model", "language", "topic_id", "statement_version", "framing_id"])
        ["canonical_score"].mean().sub(SCALE_MID).apply(np.sign)
        .unstack("framing_id")
    )
    if len(fr):
        fr["split"] = fr.apply(
            lambda r: set(r.dropna()) >= {-1.0, 1.0}, axis=1
        )
        out("\n--- D3: cross-framing direction split ---")
        out("split = some framings support and others oppose (same model x language x topic x version)")
        out(f"overall split rate = {fr['split'].mean():.3f} "
            f"(over {len(fr)} cells)")
        by_ver = fr.reset_index().groupby("statement_version")["split"].mean()
        out(by_ver.round(3).to_string())

    # ---------- save (single txt in analysis/) ----------
    outdir =  Path(outdir) if outdir else Path(__file__).resolve().parent / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    topic_ids = sorted(topics["topic_id"].astype(str))
    topic_part = "+".join(topic_ids) if len(topic_ids) <= 2 else f"{len(topic_ids)}topics"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = outdir /  f"summary_{topic_part}_{stamp}.txt"

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="CSV files from run.py")
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: the folder containing analyze.py)")
    a = p.parse_args()
    main(a.paths, a.delta, a.alpha, a.outdir)