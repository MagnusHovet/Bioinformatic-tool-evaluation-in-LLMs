"""
04_score.py
===========
Joins model recommendations against the benchmark, computes
pipeline-level scores, and engineers all features needed for
downstream statistical analysis.

Steps
-----
  1. join_benchmarks       — explode pipe-delimited tool strings and
                             merge normalised benchmark metrics.
  2. compute_pipeline_scores — aggregate tool-level scores to the
                             pipeline (model × task × prompt × run) level.
  3. add_complete_pop      — build the composite popularity index,
                             within-task binary flags, outcome variables,
                             and conflict strength.

Normalisation
-------------
  All popularity metrics (citations, stars, forks) are rank-normalised
  using the formula 1 - (r-1)/(n-1), matching the accuracy normalisation
  applied in the source benchmark (Paul et al.).

Output
------
  data/benchmark_data/ai_bioinfo_pipeline_benchmarks.csv

Usage
-----
  python 04_score.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from config.settings import RAW_DATA_PATH, SCORED_DATA_PATH, BENCHMARK_FINAL_PATH


# ── Normalisation helper ──────────────────────────────────────────────

def normalize_series(s):
    """
    Rank-based normalisation:

    Rank 1 is assigned to the highest value (ascending=False), so
    higher raw values map to scores closer to 1.  Ties receive the
    average rank.  NaNs are preserved and excluded from n.

    """
    ranks = s.rank(method="average", ascending=False, na_option="keep")
    n     = s.notna().sum()
    if n <= 1:
        return s - s
    return 1 - (ranks - 1) / (n - 1)


# ── Step 1 — Join tools against benchmark ────────────────────────────

def join_benchmarks(df_raw, df_benchmarks):
    """
    Explode pipe-delimited tool strings from the raw model output and
    left-join normalised benchmark metrics onto each tool.

    Popularity metrics (citations, stars, forks) are rank-normalised
    before the merge so all signals are on the same [0, 1] scale.

    Returns a long DataFrame with one row per tool recommendation.
    """
    # ── Explode tools ────────────────────────────────────────────────
    df_exploded         = df_raw.copy()
    df_exploded["tool"] = df_exploded["tools"].str.split("|")
    df_exploded         = df_exploded.explode("tool")
    df_exploded["tool"] = df_exploded["tool"].str.lower().str.strip()

    # ── Normalise benchmark popularity metrics ───────────────────────
    df_benchmarks = df_benchmarks.copy()
    df_benchmarks["citations_2026_norm"] = normalize_series(df_benchmarks["citations_2026"])
    df_benchmarks["stars_norm"]          = normalize_series(df_benchmarks["github_stars"])
    df_benchmarks["forks_norm"]          = normalize_series(df_benchmarks["github_forks"])

    # ── Merge ────────────────────────────────────────────────────────
    df_merged = df_exploded.merge(
        df_benchmarks[["tool", "accuracy", "citations_2026_norm", "stars_norm", "forks_norm"]],
        on="tool",
        how="left",
    )

    return df_merged


# ── Step 2 — Aggregate to pipeline level ─────────────────────────────

def compute_pipeline_scores(df_merged):
    """
    Average tool-level benchmark metrics within each pipeline
    (model × task × prompt_type × run).

    Taking the mean across tools in a pipeline treats all steps as
    equally important.  The result is one row per unique pipeline
    recommendation with accuracy and popularity scores on [0, 1].
    """
    df_scores = (
        df_merged
        .groupby(["model", "task", "prompt_type", "run"])
        .agg({
            "accuracy":           "mean",
            "citations_2026_norm": "mean",
            "stars_norm":          "mean",
            "forks_norm":          "mean",
        })
        .reset_index()
        .rename(columns={
            "accuracy":            "pipeline_accuracy",
            "citations_2026_norm": "pipeline_citations",
            "stars_norm":          "pipeline_stars",
            "forks_norm":          "pipeline_forks",
        })
    )

    return df_scores.round(5)


# ── Step 3 — Composite popularity index and outcome variables ─────────

def add_complete_pop(df):
    """
    Build the composite popularity index and all binary outcome variables
    needed for statistical analysis.
    """

    # ── Composite popularity index ───────────────────────────────────
    github_mean    = df[["pipeline_stars", "pipeline_forks"]].mean(axis=1)
    df["pop_index"] = pd.concat(
        [df["pipeline_citations"], github_mean], axis=1
    ).mean(axis=1, skipna=True)

    df["pop_index"] = df[["pipeline_citations", "pipeline_stars", "pipeline_forks"]
    ].mean(axis=1, skipna=True)

    # ── Within-task binary flags ─────────────────────────────────────
    df["high_pop"] = df.groupby("task")["pop_index"].transform(
        lambda x: (x > x.median()).astype(int)
    )
    df["high_acc"] = df.groupby("task")["pipeline_accuracy"].transform(
        lambda x: (x > x.median()).astype(int)
    )

    # ── Outcome variables ────────────────────────────────────────────
    df["pop_over_acc"]   = (df["high_pop"].astype(bool) & ~df["high_acc"].astype(bool)).astype(int)
    df["acc_over_pop"]   = (df["high_acc"].astype(bool) & ~df["high_pop"].astype(bool)).astype(int)
    df["acc_and_pop"]    = (df["high_acc"].astype(bool) &  df["high_pop"].astype(bool)).astype(int)
    df["no_acc_and_pop"] = (~df["high_acc"].astype(bool) & ~df["high_pop"].astype(bool)).astype(int)

    # ── Conflict strength ────────────────────────────────────────────
    df["conflict_strength"] = df["pop_index"] - df["pipeline_accuracy"]

    return df


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    df_raw        = pd.read_csv(RAW_DATA_PATH)
    df_benchmarks = pd.read_csv(BENCHMARK_FINAL_PATH)

    df_merged = join_benchmarks(df_raw, df_benchmarks)
    df_scores = compute_pipeline_scores(df_merged)
    df_finish = add_complete_pop(df_scores)

    os.makedirs(os.path.dirname(SCORED_DATA_PATH), exist_ok=True)
    df_finish.to_csv(SCORED_DATA_PATH, index=False)

    print(f"Saved {len(df_finish)} rows → {SCORED_DATA_PATH}")
    print(df_finish.head())