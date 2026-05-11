"""
07_sensitivity.py
=================
Sensitivity analysis for the popularity-bias metric.

Tests whether the core finding — that some models systematically recommend
popular but less accurate pipelines — holds across three different ways of
operationalising "popularity bias":

  Method 1 — Median split (primary analysis)
      A pipeline is flagged when it sits above the within-task median on
      popularity AND below the within-task median on accuracy.
      Chance baseline: 25 %.

  Method 2 — Tertile (conservative)
      Only unambiguous cases count: top third on popularity AND bottom
      third on accuracy.  Rows in the middle third of either dimension
      are excluded from the flag.  This avoids classifying borderline
      pipelines.  Chance baseline: ~11 % (1/3 × 1/3).
      Note: tertiles are computed at the unique-pipeline level to avoid
      artefacts from the high repeat rate across runs.

  Method 3 — Continuous rank gap
      pop_rank − acc_rank, where both ranks are within-task percentile
      ranks.  Positive values mean the model recommended a pipeline
      ranked higher on popularity than on accuracy.  No thresholding;
      preserves all continuous information.

If the direction and relative ordering of models is consistent across
all three methods, the primary result is robust to the choice of
operationalisation.

Output
------
  sensitivity_table.csv   — numeric results for all three methods
  fig_sensitivity.png     — three-panel figure for the paper

Usage
-----
  python 07_sensitivity.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import bootstrap

from config.settings import SCORED_DATA_PATH, FIGURES_PATH, MODELS

# ── Style ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
    "font.size":         10,
})

MODEL_ORDER = MODELS

MODEL_LABELS = {
    "anthropic/claude-sonnet-4-6": "Claude Sonnet 4.6",
    "deepseek/deepseek-chat":      "DeepSeek Chat",
    "google/gemini-2.5-pro":       "Gemini 2.5 Pro",
    "openai/gpt-5.4":              "GPT-5.4",
    "openai/gpt-5.4-mini":         "GPT-5.4 Mini",
}

MODEL_COLORS = ["#2166ac", "#4dac26", "#b8860b", "#d6604d", "#c51b7d"]


# ── Data & feature construction ───────────────────────────────────────

def load_data():
    df = pd.read_csv(SCORED_DATA_PATH)
    df = df.dropna(subset=[
        "pipeline_accuracy",
        "pipeline_citations",
        "pipeline_stars",
        "pipeline_forks",
    ])

    # Popularity index: citations vs mean GitHub signal (50/50 weight).
    # Stars and forks are combined first because they are r=0.94 correlated;
    # treating them separately would give GitHub 2/3 weight vs citations' 1/3.
    github_signal  = df[["pipeline_stars", "pipeline_forks"]].mean(axis=1)
    df["pop_index"] = pd.concat(
        [df["pipeline_citations"], github_signal], axis=1
    ).mean(axis=1)

    return df


# ── Method 1: Median split ────────────────────────────────────────────

def method_median(df):
    """
    Within each task, flag pipelines above the popularity median AND
    below the accuracy median. This is the primary analysis metric.

    Returns a copy of df with 'median_flag' added (0/1).
    """
    df = df.copy()

    df["high_pop"] = df.groupby("task")["pop_index"].transform(
        lambda x: x > x.median()
    )
    df["high_acc"] = df.groupby("task")["pipeline_accuracy"].transform(
        lambda x: x > x.median()
    )
    df["median_flag"] = (df["high_pop"] & ~df["high_acc"]).astype(int)

    return df


# ── Method 2: Tertile ─────────────────────────────────────────────────

def method_tertile(df):
    """
    Conservative version of the bias flag.

    Pipelines in the middle third of either popularity or accuracy are
    excluded — only clear cases count. This avoids the boundary
    sensitivity of the median split.

    Because models repeat the same recommendation across many runs
    (up to 95% repeat rate), computing tertiles on all rows would
    inflate the importance of the most-frequently-recommended pipeline.
    Instead, tertiles are computed on the set of *unique* pipelines
    per task, then mapped back to all rows.

    Returns a copy of df with 'tertile_flag' added (0/1 or NaN).
    NaN = pipeline falls in the excluded middle third on either
    dimension and is not counted as either biased or unbiased.
    """
    df = df.copy()
    df["tertile_flag"] = np.nan

    for task in df["task"].unique():
        task_mask = df["task"] == task
        task_rows = df[task_mask].copy()

        # Rank at unique-pipeline level to avoid repeat-rate artefacts
        unique = task_rows.drop_duplicates(
            subset=["pop_index", "pipeline_accuracy"]
        ).copy()
        unique["pop_pct"] = unique["pop_index"].rank(pct=True)
        unique["acc_pct"] = unique["pipeline_accuracy"].rank(pct=True)

        def classify(row):
            pop_hi  = row["pop_pct"] > 0.67
            pop_lo  = row["pop_pct"] <= 0.33
            acc_hi  = row["acc_pct"] > 0.67
            acc_lo  = row["acc_pct"] <= 0.33
            pop_mid = not pop_hi and not pop_lo
            acc_mid = not acc_hi and not acc_lo

            if pop_mid or acc_mid:
                return np.nan     # excluded middle — not counted
            if pop_hi and acc_lo:
                return 1.0        # unambiguous bias case
            return 0.0            # unambiguous non-bias case

        unique["flag"] = unique.apply(classify, axis=1)

        # Map unique-pipeline flags back to all rows for this task
        lookup = unique.set_index(
            ["pop_index", "pipeline_accuracy"]
        )["flag"]
        flags = task_rows.set_index(
            ["pop_index", "pipeline_accuracy"]
        ).index.map(lookup)
        df.loc[task_mask, "tertile_flag"] = flags.values

    return df


# ── Method 3: Continuous rank gap ─────────────────────────────────────

def method_rank_gap(df):
    """
    Continuous measure of popularity bias.

    For each recommendation, compute within-task percentile ranks for
    both pop_index and pipeline_accuracy, then take the difference:

        rank_gap = pop_rank - acc_rank

    Positive values: model chose a pipeline ranked higher on popularity
    than on accuracy — consistent with a popularity bias.
    Negative values: model chose a pipeline ranked higher on accuracy
    than on popularity — accuracy-aligned behaviour.

    No information is discarded to a threshold. Used as a robustness
    check: if the sign and model ordering match Methods 1 and 2, the
    primary finding is not an artefact of the threshold choice.

    Returns a copy of df with 'pop_rank', 'acc_rank', 'rank_gap' added.
    """
    df = df.copy()

    df["pop_rank"] = df.groupby("task")["pop_index"].rank(pct=True)
    df["acc_rank"] = df.groupby("task")["pipeline_accuracy"].rank(pct=True)
    df["rank_gap"] = df["pop_rank"] - df["acc_rank"]

    return df


# ── Summary table ─────────────────────────────────────────────────────

def build_summary_table(df_m1, df_m2, df_m3):
    """
    Compute per-model point estimates and 95% bootstrap CIs for each
    method. Returns a tidy DataFrame for export and printing.
    """
    rows = []

    for m in MODEL_ORDER:
        label = MODEL_LABELS[m]

        # Method 1 — median flag rate
        v1  = df_m1[df_m1["model"] == m]["median_flag"].values
        r1  = v1.mean()
        ci1 = bootstrap((v1,), np.mean, n_resamples=2000,
                        confidence_level=0.95, method="percentile")

        # Method 2 — tertile flag rate (NaNs excluded from denominator)
        v2  = df_m2[df_m2["model"] == m]["tertile_flag"].dropna().values
        r2  = v2.mean() if len(v2) > 0 else np.nan
        ci2 = (bootstrap((v2,), np.mean, n_resamples=2000,
                         confidence_level=0.95, method="percentile")
               if len(v2) > 1 else None)

        # Method 3 — mean rank gap
        v3  = df_m3[df_m3["model"] == m]["rank_gap"].values
        r3  = v3.mean()
        ci3 = bootstrap((v3,), np.mean, n_resamples=2000,
                        confidence_level=0.95, method="percentile")

        rows.append({
            "model":           label,
            "median_rate":     r1,
            "median_ci_low":   ci1.confidence_interval.low,
            "median_ci_high":  ci1.confidence_interval.high,
            "tertile_rate":    r2,
            "tertile_ci_low":  ci2.confidence_interval.low  if ci2 else np.nan,
            "tertile_ci_high": ci2.confidence_interval.high if ci2 else np.nan,
            "rank_gap_mean":   r3,
            "rank_gap_ci_low": ci3.confidence_interval.low,
            "rank_gap_ci_high":ci3.confidence_interval.high,
        })

    return pd.DataFrame(rows)


# ── Figure ────────────────────────────────────────────────────────────

def plot_sensitivity(table):
    """
    Three-panel horizontal bar / dot-and-whisker figure.

    Panel A — Median split rate (primary metric), reference at 25%.
    Panel B — Tertile rate (conservative), reference at 11% (1/3 x 1/3).
    Panel C — Mean rank gap (continuous), reference at 0.

    Models are shown in the same order across all panels so the reader
    can visually check consistency of ranking. Panel C omits y-axis
    labels since they are already shown in panels A and B.
    """

    labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
    y      = np.arange(len(MODEL_ORDER))

    fig = plt.figure(figsize=(16, 5.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.45)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    def _fmt(ax, title, xlabel, ref, ref_label, show_ylabels=True):
        ax.set_yticks(y)
        ax.set_yticklabels(labels if show_ylabels else [], fontsize=9)
        ax.axvline(ref, linestyle="--", color="black", linewidth=1.1,
                   label=ref_label)
        ax.legend(fontsize=8, frameon=False)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")

    # ── Panel A: Median split ─────────────────────────────────────────
    ax     = axes[0]
    rates  = table["median_rate"].values
    xerr_l = rates - table["median_ci_low"].values
    xerr_h = table["median_ci_high"].values - rates

    ax.barh(y, rates, xerr=[xerr_l, xerr_h],
            color=MODEL_COLORS, alpha=0.85, height=0.55,
            error_kw={"elinewidth": 1.4, "capsize": 3.5, "ecolor": "#333"})

    for i, (r, eh) in enumerate(zip(rates, xerr_h)):
        ax.text(r + eh + 0.01, i, f"{r:.1%}", va="center", fontsize=8, color="#333")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlim(0, 0.80)
    _fmt(ax,
         title="A — Median split\n(primary metric)",
         xlabel="Popularity-over-accuracy rate",
         ref=0.25, ref_label="Chance (25 %)")

    # ── Panel B: Tertile ──────────────────────────────────────────────
    ax     = axes[1]
    rates  = table["tertile_rate"].values
    xerr_l = rates - table["tertile_ci_low"].values
    xerr_h = table["tertile_ci_high"].values - rates

    ax.barh(y, rates, xerr=[xerr_l, xerr_h],
            color=MODEL_COLORS, alpha=0.85, height=0.55,
            error_kw={"elinewidth": 1.4, "capsize": 3.5, "ecolor": "#333"})

    for i, (r, eh) in enumerate(zip(rates, xerr_h)):
        if not np.isnan(r):
            ax.text(r + eh + 0.005, i, f"{r:.1%}", va="center", fontsize=8, color="#333")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlim(0, 0.70)
    _fmt(ax,
         title="B — Tertile\n(conservative, middle excluded)",
         xlabel="Popularity-over-accuracy rate",
         ref=0.111, ref_label="Chance (11 %)")

    # ── Panel C: Rank gap ─────────────────────────────────────────────
    ax     = axes[2]
    gaps   = table["rank_gap_mean"].values
    xerr_l = gaps - table["rank_gap_ci_low"].values
    xerr_h = table["rank_gap_ci_high"].values - gaps

    ax.errorbar(gaps, y,
                xerr=[xerr_l, xerr_h],
                fmt="none", elinewidth=1.6, capsize=4, ecolor="#555",
                zorder=3)

    for i, (gap, color) in enumerate(zip(gaps, MODEL_COLORS)):
        ax.scatter(gap, i, color=color, s=70, zorder=4)

    _fmt(ax,
         title="C — Rank gap\n(continuous, no threshold)",
         xlabel="Mean (pop rank − acc rank)",
         ref=0.0, ref_label="No bias (0)",
         show_ylabels=False)

    fig.suptitle(
        "Sensitivity analysis — consistency of popularity bias finding\n"
        "across three operationalisations of the outcome variable",
        fontsize=11, fontweight="bold", x=0.01, ha="left",
    )

    fig.subplots_adjust(top=0.85, wspace=0.45)
    path = os.path.join(FIGURES_PATH, "fig_sensitivity.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── Console report ────────────────────────────────────────────────────

def print_report(table):
    """
    Print a structured summary to stdout including a consistency check:
    do all three methods agree on which models are above vs. below their
    respective chance baselines?
    """

    MEDIAN_NULL  = 0.25
    TERTILE_NULL = 0.111
    RANK_NULL    = 0.0

    print("\n" + "=" * 72)
    print("  SENSITIVITY ANALYSIS — Popularity Bias Across Three Methods")
    print("=" * 72)

    header = (f"{'Model':<25} {'Median':>8} {'Tertile':>8} "
              f"{'RankGap':>9}  {'Consistent?':>12}")
    print(header)
    print("-" * 72)

    for _, row in table.iterrows():
        m1_above = row["median_rate"]   > MEDIAN_NULL
        m2_above = row["tertile_rate"]  > TERTILE_NULL
        m3_above = row["rank_gap_mean"] > RANK_NULL
        consistent = "YES" if (m1_above == m2_above == m3_above) else "NO ⚠"

        print(
            f"{row['model']:<25} "
            f"{row['median_rate']:>8.1%} "
            f"{row['tertile_rate']:>8.1%} "
            f"{row['rank_gap_mean']:>+9.3f}  "
            f"{consistent:>12}"
        )

    print("-" * 72)
    print(f"\nChance baselines:  Median={MEDIAN_NULL:.0%}  "
          f"Tertile={TERTILE_NULL:.1%}  RankGap={RANK_NULL:.1f}")

    n_consistent = sum(
        (r["median_rate"] > MEDIAN_NULL) ==
        (r["tertile_rate"] > TERTILE_NULL) ==
        (r["rank_gap_mean"] > RANK_NULL)
        for _, r in table.iterrows()
    )

    print(f"\n  {n_consistent}/{len(table)} models consistent across all three methods.")
    print("\nInterpretation:")
    print("  YES → finding robust to operationalisation choice.")
    print("  NO  → result depends on threshold; interpret with caution.")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(FIGURES_PATH, exist_ok=True)

    print("\nLoading data...")
    df = load_data()
    print(f"  {len(df)} rows | {df['model'].nunique()} models | "
          f"{df['task'].nunique()} tasks")

    print("\nComputing methods...")
    df_m1 = method_median(df)
    df_m2 = method_tertile(df)
    df_m3 = method_rank_gap(df)
    print("  Done.")

    print("\nBuilding summary table...")
    table = build_summary_table(df_m1, df_m2, df_m3)

    out_path = os.path.join(os.path.dirname(FIGURES_PATH), "sensitivity_table.csv")
    table.round(4).to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    print_report(table)

    print("\nGenerating figure...")
    plot_sensitivity(table)

    print("\nDone.")