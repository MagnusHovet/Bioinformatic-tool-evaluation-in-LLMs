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
    BENCHMARK_FILTERED_PATH
)

# ─────────────────────────────────────────────────────────────────────
# 0. BUILD BENCHMARK FROM RAW TSVs
# ─────────────────────────────────────────────────────────────────────

def build_benchmark():

    print("\n[Step 0] Building tool benchmark from ACC + CIT")

    acc = pd.read_csv(ACC_DATA_PATH, sep="\t")
    cit = pd.read_csv(CIT_DATA_PATH, sep="\t")

    # ── Clean names ──────────────────────────────────────────────────
    acc["method"] = acc["method"].str.strip().str.lower()
    cit["tool"]   = cit["tool"].str.strip().str.lower()

    # ── Aggregate accuracy (mean per tool) ───────────────────────────
    acc_agg = (
        acc.groupby("method")["accuracyRank"]
        .mean()
        .reset_index()
        .rename(columns={"method": "tool", "accuracyRank": "accuracy"})
    )

    # ── Extract citations (use most recent available) ────────────────
    # Prefer 2020, fallback to 2017
    cit["citations"] = cit["totalCitations(2020)"].fillna(
        cit["totalCitations(2017)"]
    )

    cit_agg = cit[["tool", "citations"]].copy()

    # ── Merge ───────────────────────────────────────────────────────
    benchmark = acc_agg.merge(cit_agg, on="tool", how="outer")

    # ── Clean types ─────────────────────────────────────────────────
    benchmark["citations"] = pd.to_numeric(
        benchmark["citations"], errors="coerce"
    )

    # ── Save ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(BENCHMARK_PATH), exist_ok=True)
    benchmark.to_csv(BENCHMARK_PATH, index=False)

    print(f"Benchmark created: {len(benchmark)} tools")
    print(f"Saved → {BENCHMARK_PATH}")

    return benchmark


# ─────────────────────────────────────────────────────────────────────
# FILTER TOOLS USED BY MODELS
# ─────────────────────────────────────────────────────────────────────

def extract_tools(tool_string):
    if pd.isna(tool_string):
        return []
    return [t.strip().lower() for t in str(tool_string).split("|") if t.strip()]


def filter_benchmark():

    df = pd.read_csv(RAW_DATA_PATH)
    benchmark = pd.read_csv(BENCHMARK_PATH)

    model_tools = set()

    for row in df["tools"]:
        model_tools.update(extract_tools(row))

    model_tools = sorted(model_tools)
    
    benchmark["tool"] = benchmark["tool"].str.strip().str.lower()
    filtered = benchmark[benchmark["tool"].isin(model_tools)].copy()
    
    missing = sorted(set(model_tools) - set(filtered["tool"]))
    if missing:
        print("Example missing tools:", missing[:10])

    os.makedirs(os.path.dirname(BENCHMARK_FILTERED_PATH), exist_ok=True)
    filtered.to_csv(BENCHMARK_FILTERED_PATH, index=False)
    
    print(f"Tools used by models: {len(model_tools)}")
    print(f"Tools kept in benchmark: {len(filtered)}")
    print(f"Missing tools (not in benchmark): {len(missing)}")

    print("Saved → tool_benchmark_filtered.csv")

    return filtered


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    benchmark = build_benchmark()
    filtered  = filter_benchmark()

    
    print("\nDone.")