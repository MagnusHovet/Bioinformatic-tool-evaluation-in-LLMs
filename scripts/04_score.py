import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from config.settings import RAW_DATA_PATH, SCORED_DATA_PATH, BENCHMARK_FINAL_PATH


# =========================
# NORMALIZATION HELPERS
# =========================

def normalize_series(s):
    ranks = s.rank(method='average', ascending=False, na_option='keep')
    n = s.notna().sum()
    if n <= 1:
        return s - s
    return 1 - (ranks - 1) / (n - 1)

# =========================
# --- 1. JOIN TOOLS AGAINST BENCHMARK ---
# =========================

def join_benchmarks(df_normalized, df_benchmarks):

    # explode tools
    df_exploded = df_normalized.copy()
    df_exploded["tool"] = df_exploded["tools"].str.split("|")
    df_exploded = df_exploded.explode("tool")
    df_exploded["tool"] = df_exploded["tool"].str.lower().str.strip()

    # ---  preprocess GitHub metrics ---
    df_benchmarks = df_benchmarks.copy()

    # normalize to 0–1
    df_benchmarks["citations_2026_norm"] = normalize_series(df_benchmarks["citations_2026"])
    df_benchmarks["stars_norm"] = normalize_series(df_benchmarks["github_stars"])
    df_benchmarks["forks_norm"] = normalize_series(df_benchmarks["github_forks"])

    # merge
    df_merged = df_exploded.merge(
        df_benchmarks[
            ["tool", "accuracy", "citations_2026_norm", "stars_norm", "forks_norm"]
        ],
        on="tool",
        how="left"
    )

    return df_merged


# =========================
# --- 2. COMPUTE PIPELINE-LEVEL SCORES ---
# =========================

def compute_pipeline_scores(df_merged):
    
    df_scores = (
        df_merged
        .groupby(["model", "task", "prompt_type", "run"])
        .agg({
            "accuracy": "mean",
            "citations_2026_norm": "mean",
            "stars_norm": "mean",
            "forks_norm": "mean"
        })
        .reset_index()
        .rename(columns={
            "accuracy": "pipeline_accuracy",
            "citations_2026_norm": "pipeline_citations",
            "stars_norm": "pipeline_stars",
            "forks_norm": "pipeline_forks"
        })
    )

    df_scores = df_scores.round(5)

    return df_scores

# =========================
# --- 3. Complete popularity metric ---
# =========================
def add_complete_pop(df):
    
    df = df.dropna(subset=[
        "pipeline_accuracy",
        "pipeline_citations",
        "pipeline_stars",
        "pipeline_forks",
    ])

    # All signals are pre-normalised to 0–1; simple mean preserves that scale
    github_mean = df[["pipeline_stars", "pipeline_forks"]].mean(axis=1)
    df["pop_index"] = pd.concat([df["pipeline_citations"], github_mean], axis=1).mean(axis=1)

    df["high_pop"] = df.groupby("task")["pop_index"].transform(
        lambda x: x > x.median()
    )
    df["high_acc"] = df.groupby("task")["pipeline_accuracy"].transform(
        lambda x: x > x.median()
    )

    df["pop_over_acc"] = (df["high_pop"] & ~df["high_acc"]).astype(int)
    df["conflict_strength"] = df["pop_index"] - df["pipeline_accuracy"]

    return df

# =========================
# --- 3. MAIN ---
# =========================

if __name__ == "__main__":

    df_normalized = pd.read_csv(RAW_DATA_PATH)
    df_benchmarks = pd.read_csv(BENCHMARK_FINAL_PATH)

    df_merged = join_benchmarks(df_normalized, df_benchmarks)
    df_scores = compute_pipeline_scores(df_merged)
    df_finish = add_complete_pop(df_scores)

    os.makedirs(os.path.dirname(SCORED_DATA_PATH), exist_ok=True)
    df_finish.to_csv(SCORED_DATA_PATH, index=False)

    print(df_finish.head())