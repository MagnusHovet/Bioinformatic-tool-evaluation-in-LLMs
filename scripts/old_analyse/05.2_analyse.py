import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from scipy.stats import bootstrap
from sklearn.preprocessing import StandardScaler

from config.settings import SCORED_DATA_PATH, FIGURES_PATH

# ── Style ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.size": 10,
})

def _save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, name), bbox_inches="tight")
    plt.close()
    print(f"Saved: {name}")

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

TASK_ORDER = ["Phylogeny", "RNA-seq", "genome_assembly", "variant_detection"]

TASK_LABELS = {
    "Phylogeny":         "Phylogeny",
    "RNA-seq":           "RNA-seq",
    "genome_assembly":   "Genome assembly",
    "variant_detection": "Variant detection",
}

COLORS = ["#2166ac", "#4dac26", "#b8860b", "#d6604d", "#c51b7d"]

# ── Data ──────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(SCORED_DATA_PATH)

    df = df.dropna(subset=[
        "pipeline_accuracy",
        "pipeline_citations",
        "pipeline_stars",
        "pipeline_forks",
    ])

    # unified popularity index (simple mean of normalized metrics)
    df["pop_index"] = df[["pipeline_citations", "pipeline_stars", "pipeline_forks"]].mean(axis=1)

    # popularity bias = how much popularity exceeds accuracy
    df["pop_bias"] = df["pop_index"] - df["pipeline_accuracy"]

    # conflict classification: within each task, flag high-pop / high-acc pipelines
    df["high_pop"] = df.groupby("task")["pop_index"].transform(
        lambda x: x > x.median()
    )
    df["high_acc"] = df.groupby("task")["pipeline_accuracy"].transform(
        lambda x: x > x.median()
    )

    # binary flag: model chose a popular but less accurate pipeline
    df["pop_over_acc"] = (df["high_pop"] & ~df["high_acc"]).astype(int)

    return df

# ──────────────────────────────────────────────────────────────────────
# FIGURE 1 — POPULARITY BIAS RATE PER MODEL
# ──────────────────────────────────────────────────────────────────────

def fig1_bias_rate(df):
    """
    For each model, the rate at which it recommended a popular but
    less accurate pipeline (with bootstrap 95% CI).
    Reference line at 25% = expected rate by chance.
    """

    rates, ci_lows, ci_highs = [], [], []

    for m in MODEL_ORDER:
        vals = df[df["model"] == m]["pop_over_acc"].values
        rate = vals.mean()
        res  = bootstrap(
            (vals,), np.mean,
            n_resamples=2000,
            confidence_level=0.95,
            method="percentile",
        )
        rates.append(rate)
        ci_lows.append(rate - res.confidence_interval.low)
        ci_highs.append(res.confidence_interval.high - rate)

    labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
    y      = np.arange(len(MODEL_ORDER))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.barh(
        y, rates,
        xerr=[ci_lows, ci_highs],
        color=COLORS, alpha=0.85, height=0.55,
        error_kw={"elinewidth": 1.5, "capsize": 4, "ecolor": "#333333"},
    )

    ax.axvline(0.25, linestyle="--", color="black", linewidth=1.2,
               label="Expected by chance (25%)")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Rate of popularity-over-accuracy recommendations", fontsize=10)
    ax.set_xlim(0, 0.75)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=9, frameon=False)
    ax.set_title(
        "Figure 1 — How often does each model recommend\n"
        "a popular but less accurate pipeline?",
        fontsize=11, fontweight="bold", loc="left",
    )

    # annotate exact rates
    for i, (r, ch) in enumerate(zip(rates, ci_highs)):
        ax.text(r + ch + 0.02, i, f"{r:.1%}", va="center", fontsize=9, color="#333")

    _save("fig1_bias_rate.png")

# ──────────────────────────────────────────────────────────────────────
# FIGURE 2 — ACCURACY vs POPULARITY BY MODEL AND PROMPT TYPE
# ──────────────────────────────────────────────────────────────────────

def fig2_acc_vs_pop(df):
    """
    Paired bar chart showing mean pipeline accuracy vs mean popularity
    index per model, split by prompt type (expert / naive).
    Arrows indicate the direction of the gap.
    """

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    for ax, ptype, title in zip(
        axes,
        ["expert", "naive"],
        ["Expert prompt", "Naive prompt"],
    ):
        sub   = df[df["prompt_type"] == ptype]
        means = sub.groupby("model")[["pipeline_accuracy", "pop_index"]].mean()

        x     = np.arange(len(MODEL_ORDER))
        width = 0.35

        acc_vals = [means.loc[m, "pipeline_accuracy"] for m in MODEL_ORDER]
        pop_vals = [means.loc[m, "pop_index"]         for m in MODEL_ORDER]

        ax.bar(x - width / 2, acc_vals, width,
               label="Pipeline accuracy", color="#2166ac", alpha=0.85)
        ax.bar(x + width / 2, pop_vals, width,
               label="Popularity index",  color="#d6604d", alpha=0.85)

        # arrows showing the accuracy → popularity gap
        for i, (a, p) in enumerate(zip(acc_vals, pop_vals)):
            ax.annotate(
                "", xy=(i + width / 2, p), xytext=(i - width / 2, a),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=0.8),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_LABELS[m].replace(" ", "\n") for m in MODEL_ORDER],
            fontsize=8.5,
        )
        ax.set_ylim(0.55, 0.95)
        ax.set_title(title, fontsize=11, fontweight="bold")

    axes[0].set_ylabel("Normalised score (0–1)", fontsize=10)
    axes[0].legend(fontsize=9, frameon=False)

    fig.suptitle(
        "Figure 2 — What models recommend: accuracy vs popularity of suggested pipelines",
        fontsize=11, fontweight="bold", x=0.05, ha="left",
    )

    _save("fig2_acc_vs_pop.png")

# ──────────────────────────────────────────────────────────────────────
# FIGURE 3 — BIAS RATE HEATMAP BY MODEL × TASK
# ──────────────────────────────────────────────────────────────────────

def fig3_bias_heatmap(df):
    """
    Heatmap of popularity-over-accuracy rate for each model × task
    combination. Red = frequent popularity bias; green = accuracy-aligned.
    """

    pivot = df.groupby(["model", "task"])["pop_over_acc"].mean().unstack()
    pivot = pivot.loc[MODEL_ORDER, TASK_ORDER]
    pivot.index   = [MODEL_LABELS[m] for m in MODEL_ORDER]
    pivot.columns = [TASK_LABELS[t]  for t in TASK_ORDER]

    fig, ax = plt.subplots(figsize=(7, 4))

    im = ax.imshow(pivot.values, cmap="RdYlGn_r", vmin=0, vmax=0.7, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val   = pivot.values[i, j]
            color = "white" if val > 0.45 else "black"
            ax.text(j, i, f"{val:.0%}",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Rate of pop-over-acc\nrecommendations", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )

    ax.set_title(
        "Figure 3 — Popularity bias rate by model and task\n"
        "(red = model frequently chose popular over accurate)",
        fontsize=11, fontweight="bold", loc="left",
    )

    _save("fig3_bias_heatmap.png")

# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(FIGURES_PATH, exist_ok=True)

    df = load_data()

    print("Generating 3-figure publication pipeline...\n")

    fig1_bias_rate(df)
    fig2_acc_vs_pop(df)
    fig3_bias_heatmap(df)

    print("\nDone.")