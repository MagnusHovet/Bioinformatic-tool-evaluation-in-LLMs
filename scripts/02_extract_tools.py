"""
02_extract_tools.py
===================
Builds the tool benchmark from raw TSV files and filters it to only
the tools actually recommended by models in the collection step.

Steps
-----
  1. build_benchmark  — merge accuracy ranks and citation counts from
                        Paul et al. into a single benchmark CSV.
  2. filter_benchmark — retain only tools that appear in the raw
                        model output, and report any missing tools.

Output
------
  data/benchmarks/tool_benchmark.csv
  data/benchmarks/tool_benchmark_filtered.csv

Usage
-----
  python 02_extract_tools.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from config.settings import (
    RAW_DATA_PATH,
    ACC_DATA_PATH,
    CIT_DATA_PATH,
    BENCHMARK_PATH,
    BENCHMARK_FILTERED_PATH,
)


# ── Step 1 — Build benchmark from raw TSVs ────────────────────────────

def build_benchmark():
    """
    Merge accuracy ranks and citation counts from the Paul et al. TSVs
    into a single benchmark table.

    Accuracy is averaged across all benchmark entries per tool
    (mean accuracyRank).  Citations prefer the 2020 count and fall
    back to 2017 where 2020 is missing.

    Returns the merged benchmark DataFrame and saves it to BENCHMARK_PATH.
    """
    print("\n[Step 1] Building tool benchmark from ACC + CIT")

    acc = pd.read_csv(ACC_DATA_PATH, sep="\t")
    cit = pd.read_csv(CIT_DATA_PATH, sep="\t")

    # ── Clean names ──────────────────────────────────────────────────
    acc["method"] = acc["method"].str.strip().str.lower()
    cit["tool"]   = cit["tool"].str.strip().str.lower()

    # ── Aggregate accuracy: mean rank per tool ───────────────────────
    acc_agg = (
        acc.groupby("method")["accuracyRank"]
        .mean()
        .reset_index()
        .rename(columns={"method": "tool", "accuracyRank": "accuracy"})
    )

    # ── Extract citations: prefer 2020, fall back to 2017 ───────────
    cit["citations"] = cit["totalCitations(2020)"].fillna(
        cit["totalCitations(2017)"]
    )
    cit_agg = cit[["tool", "citations"]].copy()

    # ── Merge and clean types ────────────────────────────────────────
    benchmark = acc_agg.merge(cit_agg, on="tool", how="outer")
    benchmark["citations"] = pd.to_numeric(benchmark["citations"], errors="coerce")

    # ── Save ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(BENCHMARK_PATH), exist_ok=True)
    benchmark.to_csv(BENCHMARK_PATH, index=False)

    print(f"  Benchmark created: {len(benchmark)} tools")
    print(f"  Saved → {BENCHMARK_PATH}")

    return benchmark


# ── Step 2 — Filter to model-recommended tools ────────────────────────

def extract_tools(tool_string):
    """Parse a pipe-delimited tool string into a list of clean tool names."""
    if pd.isna(tool_string):
        return []
    return [t.strip().lower() for t in str(tool_string).split("|") if t.strip()]


def filter_benchmark():
    """
    Identify the unique set of tools recommended by models in the raw
    output, then filter the benchmark to those tools only.

    Reports tools recommended by models that are missing from the
    benchmark (i.e. tools that cannot be scored).

    Returns the filtered benchmark DataFrame and saves it to
    BENCHMARK_FILTERED_PATH.
    """
    print("\n[Step 2] Filtering benchmark to model-recommended tools")

    df        = pd.read_csv(RAW_DATA_PATH)
    benchmark = pd.read_csv(BENCHMARK_PATH)

    # ── Collect all unique tools recommended by models ───────────────
    model_tools = set()
    for row in df["tools"]:
        model_tools.update(extract_tools(row))
    model_tools = sorted(model_tools)

    # ── Filter benchmark ─────────────────────────────────────────────
    benchmark["tool"] = benchmark["tool"].str.strip().str.lower()
    filtered          = benchmark[benchmark["tool"].isin(model_tools)].copy()

    missing = sorted(set(model_tools) - set(filtered["tool"]))
    if missing:
        print(f"  Example missing tools: {missing[:10]}")

    # ── Save ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(BENCHMARK_FILTERED_PATH), exist_ok=True)
    filtered.to_csv(BENCHMARK_FILTERED_PATH, index=False)

    print(f"  Tools used by models:          {len(model_tools)}")
    print(f"  Tools kept in benchmark:       {len(filtered)}")
    print(f"  Missing tools (not in bench):  {len(missing)}")
    print(f"  Saved → {BENCHMARK_FILTERED_PATH}")

    return filtered


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    benchmark = build_benchmark()
    filtered  = filter_benchmark()

    print("\nDone.")