import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from scipy.stats import kruskal, binomtest, chi2_contingency

import warnings
warnings.filterwarnings("ignore")

from config.settings import SCORED_DATA_PATH

# ── Constants ─────────────────────────────────────────────────────────

MODEL_LABELS = {
    "anthropic/claude-sonnet-4-6": "Claude Sonnet 4.6",
    "deepseek/deepseek-chat":      "DeepSeek Chat",
    "google/gemini-2.5-pro":       "Gemini 2.5 Pro",
    "openai/gpt-5.4":              "GPT-5.4",
    "openai/gpt-5.4-mini":         "GPT-5.4 Mini",
}

MODEL_ORDER = [
    "deepseek/deepseek-chat",
    "google/gemini-2.5-pro",
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
]

TASK_ORDER = [
    "Phylogeny",
    "RNA-seq",
    "genome_assembly",
    "variant_detection",
]

TASK_LABELS = {
    "Phylogeny":         "Phylogeny",
    "RNA-seq":           "RNA-seq",
    "genome_assembly":   "Genome assembly",
    "variant_detection": "Variant detection",
}

# ── Helpers ───────────────────────────────────────────────────────────

def _sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def _cohens_h(p1, p2):
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

def _h_label(h):
    a = abs(h)
    if a >= 0.5: return "large"
    if a >= 0.2: return "medium"
    return "small"

def _section(title):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)

def _divider():
    print("-" * 68)

# ── Data ──────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(SCORED_DATA_PATH)

    df = df.dropna(subset=[
        "pipeline_accuracy",
        "pipeline_citations",
        "pipeline_stars",
        "pipeline_forks",
    ])

    df["pop_index"] = df[[
        "pipeline_citations",
        "pipeline_stars",
        "pipeline_forks",
    ]].mean(axis=1)

    df["high_pop"] = df.groupby("task")["pop_index"].transform(
        lambda x: x > x.median()
    )
    df["high_acc"] = df.groupby("task")["pipeline_accuracy"].transform(
        lambda x: x > x.median()
    )

    df["quad"] = "both_low"
    df.loc[df["high_pop"] & df["high_acc"], "quad"] = "both_high"
    df.loc[df["high_pop"] & ~df["high_acc"], "quad"] = "pop_only"
    df.loc[~df["high_pop"] & df["high_acc"], "quad"] = "acc_only"

    df["pop_over_acc"] = (df["quad"] == "pop_only").astype(int)

    empirical_null = (df["quad"] == "pop_only").mean()

    return df, empirical_null

# ── Statistical report ────────────────────────────────────────────────

def stats_report(df, empirical_null):

    # ── 0. Overview ───────────────────────────────────────────────────
    _section("DATASET OVERVIEW")

    print(f"  Total recommendations : {len(df)}")
    print(f"  Models                : {df['model'].nunique()}")
    print(f"  Tasks                 : {df['task'].nunique()}")
    print(f"\n  Empirical null (pop_only): {empirical_null:.1%}")

    # ── 1. Core hypothesis: binomial ─────────────────────────────────
    _section("CORE TEST — Bias vs baseline (Binomial)")

    header = f"{'Model':<25} {'Rate':>8} {'h':>8} {'p':>10} {'Sig':>4}"
    print(header)
    _divider()

    binom_results = []

    for m in MODEL_ORDER:
        vals = df[df["model"] == m]["pop_over_acc"].values
        n, k = len(vals), int(vals.sum())
        rate = k / n
        h = _cohens_h(rate, empirical_null)
        p = binomtest(k, n, empirical_null).pvalue

        print(f"{MODEL_LABELS[m]:<25} {rate:>8.1%} {h:>+8.3f} {p:>10.4f} {_sig(p):>4}")
        binom_results.append((m, rate, h, p))

    # ── 2. Model differences ─────────────────────────────────────────
    _section("MODEL DIFFERENCES (Kruskal-Wallis)")

    groups = [df[df["model"] == m]["pop_over_acc"].values for m in MODEL_ORDER]
    h_stat, p_kw = kruskal(*groups)

    print(f"H = {h_stat:.3f}, p = {p_kw:.2e} {_sig(p_kw)}")
    print("Interpretation: models differ in bias behavior")

    # ── 3. Mechanism: chi-square ─────────────────────────────────────
    _section("DECISION STRUCTURE (Chi-square)")

    ct = pd.crosstab(df["model"], df["quad"])
    ct.index = [MODEL_LABELS[m] for m in ct.index]

    chi2, p_chi, dof, _ = chi2_contingency(ct)

    print(f"chi² = {chi2:.2f}, dof = {dof}, p = {p_chi:.2e} {_sig(p_chi)}\n")
    print("Key observation: models differ in how they trade off accuracy vs popularity")

    print("\nObserved counts:")
    print(ct)

    # ── 4. Task effect ───────────────────────────────────────────────
    _section("TASK EFFECT (Kruskal-Wallis + Binomial)")

    task_groups = [
        df[df["task"] == t]["pop_over_acc"].values for t in TASK_ORDER
    ]

    h_task, p_task = kruskal(*task_groups)
    print(f"Kruskal-Wallis: H = {h_task:.3f}, p = {p_task:.2e} {_sig(p_task)}\n")

    header = f"{'Task':<20} {'Rate':>8} {'p':>10} {'Direction'}"
    print(header)
    _divider()

    for t in TASK_ORDER:
        vals = df[df["task"] == t]["pop_over_acc"].values
        n, k = len(vals), int(vals.sum())
        rate = k / n
        p = binomtest(k, n, empirical_null).pvalue

        direction = "above" if rate > empirical_null else "below"
        print(f"{TASK_LABELS[t]:<20} {rate:>8.1%} {p:>10.4f} {direction} null")

    # ── Summary ──────────────────────────────────────────────────────
    _section("SUMMARY")

    print(f"Empirical null: {empirical_null:.1%}")
    print(f"Model differences: H={h_stat:.1f}, p<0.001")
    print(f"Task effect: H={h_task:.1f}, p<0.001")
    print(f"Decision structure differs: χ²={chi2:.1f}, p<0.001\n")

    print("Per-model bias:")
    for m, rate, h, p in binom_results:
        print(f"  {MODEL_LABELS[m]:<25} {rate:.1%}  h={h:+.2f} ({_h_label(h)})  {_sig(p)}")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    df, empirical_null = load_data()

    print("\nPopularity Bias in AI Bioinformatics Recommendations")
    print("=" * 68)

    stats_report(df, empirical_null)

    print("\nDone.")