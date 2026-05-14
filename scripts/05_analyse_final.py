"""
05_analyse_final.py
===================
Main publication figures for:
  "Popularity vs accuracy in AI-generated bioinformatic pipeline recommendations"

Figures
-------
  Fig 1 — Distribution of all four recommendation outcome categories
          (A = by model, B = by task)
  Fig 2 — Bias-rate heatmap by model × task
          (A–D = one panel per binary outcome)
  Fig 3 — Statistical evidence for popularity bias and between-model
          heterogeneity (A = two-sided binomial, B = permutation test)
  Fig 4 — Popularity signals vs pipeline accuracy (Spearman scatter)

Usage
-----
  python 05_analyse_final.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, bootstrap, binomtest

from config.settings import SCORED_DATA_PATH, FIGURES_PATH, MODELS, TASKS


# ── Style ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
    "font.size":         10,
})

MODEL_ORDER = MODELS
TASK_ORDER  = list(TASKS.keys())

MODEL_LABELS = {
    "anthropic/claude-sonnet-4-6": "Claude Sonnet 4.6",
    "deepseek/deepseek-chat":      "DeepSeek Chat",
    "google/gemini-2.5-pro":       "Gemini 2.5 Pro",
    "openai/gpt-5.4":              "GPT-5.4",
    "openai/gpt-5.4-mini":         "GPT-5.4 Mini",
}

TASK_LABELS = {
    "Phylogeny":         "Phylogeny",
    "RNA-seq":           "RNA-seq",
    "genome_assembly":   "Genome assembly",
    "variant_detection": "Variant detection",
}

MODEL_COLORS = ["#2166ac", "#4dac26", "#b8860b", "#d6604d", "#c51b7d"]
ACC_COLOR    = "#2166ac"
POP_COLOR    = "#d6604d"

CHANCE = 0.25


# ── Helpers ───────────────────────────────────────────────────────────

def _save(name):
    """Save the current figure to FIGURES_PATH and close it."""
    plt.tight_layout()
    path = os.path.join(FIGURES_PATH, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def _sig_marker(p, rate, chance=CHANCE):
    """
    Return a significance marker encoding both significance level and
    direction relative to the chance baseline.

    Above chance: ↑* / ↑** / ↑***
    Below chance: ↓* / ↓** / ↓***
    Not significant: ns
    """
    if p >= 0.05:
        return "ns"
    direction = "↑" if rate > chance else "↓"
    if p < 0.001:
        return f"{direction}***"
    elif p < 0.01:
        return f"{direction}**"
    else:
        return f"{direction}*"


def load_data():
    """Load scored pipeline data from SCORED_DATA_PATH."""
    return pd.read_csv(SCORED_DATA_PATH)


# ── Fig 1 — Recommendation outcome distribution (A=model, B=task) ─────

def fig1_recommendation_distribution(df):
    """
    Two-panel stacked horizontal bar chart showing all four binary
    outcome proportions.

    Panel A — grouped by model.
    Panel B — grouped by task.
    """
    cols   = ["pop_over_acc", "acc_over_pop", "acc_and_pop", "no_acc_and_pop"]
    labels = [                      # shortened for legend readability
        "Pop > Acc",
        "Acc > Pop",
        "High Acc/Pop",
        "Low Acc/Pop",
    ]
    colors = ["#E24B4A", "#D8D509", "#2514DC", "#1F8EB6"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    panels = [
        ("model", MODEL_ORDER, [MODEL_LABELS[m] for m in MODEL_ORDER], "A "),
        ("task",  TASK_ORDER,  [TASK_LABELS[t]  for t in TASK_ORDER],  "B "),
    ]

    for ax, (group_by, order, bar_labels, panel_title) in zip(axes, panels):
        means   = df.groupby(group_by)[cols].mean().mul(100).reindex(order)
        bottoms = np.zeros(len(order))

        for col, label, color in zip(cols, labels, colors):
            vals = means[col].values
            ax.barh(
                bar_labels, vals,
                left=bottoms,
                color=color,
                label=label,
                height=0.6,
            )
            for i, (v, b) in enumerate(zip(vals, bottoms)):
                if v > 6:
                    ax.text(
                        b + v / 2, i, f"{v:.0f}%",
                        ha="center", va="center",
                        fontsize=9, color="white", fontweight="500",
                    )
            bottoms += vals

        ax.set_xlabel("Proportion of recommendations (%)", fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_title(panel_title, fontsize=10, fontweight="bold", loc="left")

    axes[1].legend(
        fontsize=8, frameon=False,
        bbox_to_anchor=(1.01, 1), loc="upper left",
        handlelength=1.2, handleheight=0.8,   # smaller legend handles
    )

    fig.suptitle(
        "Distribution of pipeline recommendation outcomes",
        fontsize=11, fontweight="bold", x=0.01, ha="left",
    )

    _save("fig1_recommendation_distribution.png")


# ── Fig 2 — Bias heatmap (A–D, one per binary outcome) ───────────────

def fig2_bias_heatmap_combined(df):
    """
    2×2 grid of heatmaps, one per binary outcome category.

    Panel A — popularity over accuracy  (pop_over_acc)
    Panel B — accuracy over popularity  (acc_over_pop)
    Panel C — high accuracy & popularity (acc_and_pop)
    Panel D — low accuracy & popularity  (no_acc_and_pop)

    Y-axis labels (models) shown on left column only.
    X-axis labels (tasks) shown on bottom row only.
    One shared colorbar on the right.
    """
    outcomes = [
        ("pop_over_acc",   "A — Popularity over accuracy"),
        ("acc_over_pop",   "B — Accuracy over popularity"),
        ("acc_and_pop",    "C — High accuracy & popularity"),
        ("no_acc_and_pop", "D — Low accuracy & popularity"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    im_last = None
    for idx, (ax, (outcome, panel_title)) in enumerate(zip(axes.flatten(), outcomes)):
        row = idx // 2
        col = idx  % 2

        pivot = (
            df.groupby(["model", "task"])[outcome]
            .mean()
            .unstack()
        )
        pivot         = pivot.loc[MODEL_ORDER, TASK_ORDER]
        pivot.index   = [MODEL_LABELS[m] for m in MODEL_ORDER]
        pivot.columns = [TASK_LABELS[t]  for t in TASK_ORDER]

        im_last = ax.imshow(
            pivot.values, cmap="viridis", vmin=0, vmax=0.7, aspect="auto"
        )

        # ── X-axis: only on bottom row ────────────────────────────────
        if row == 1:
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, fontsize=9)
        else:
            ax.set_xticks([])

        # ── Y-axis: only on left column ───────────────────────────────
        if col == 0:
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=9)
        else:
            ax.set_yticks([])

        ax.set_title(panel_title, fontsize=10, fontweight="bold", loc="left")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val   = pivot.values[i, j]
                color = "black" if val > 0.7 else "white"
                ax.text(j, i, f"{val:.0%}",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    # ── Single shared colorbar ────────────────────────────────────────

    fig.suptitle(
        "Recommendation outcome rates by model and task",
        fontsize=11, fontweight="bold", x=0.01, ha="left",
    )

    _save("fig2_bias_heatmap_combined.png")


# ── Fig 3 — Binomial test + permutation test ──────────────────────────

def fig3_bias_significance(df):
    """
    Two-part significance test for popularity bias.

    Panel A — Per-model two-sided binomial test against the 25% chance
        baseline.  Two-sided testing captures models significantly below
        chance (Gemini, DeepSeek), which represent genuine accuracy-seeking
        behaviour rather than an absence of bias.  Direction is encoded in
        the significance marker (↑ = above chance, ↓ = below chance).

    Panel B — Permutation test on between-model variance in bias rates.
        Shuffles model labels 10,000 times to build a null distribution
        of variance.  Tests whether the observed spread across models is
        larger than chance — i.e. whether models genuinely differ in their
        susceptibility to popularity bias.  Log scale on x-axis reveals
        the full shape of the null distribution.
    """
    N_PERM = 10_000

    # ── Part A: per-model two-sided binomial tests ────────────────────
    results = []

    for m in MODEL_ORDER:
        vals = df[df["model"] == m]["pop_over_acc"].values
        n    = len(vals)
        k    = int(vals.sum())
        rate = k / n
        test = binomtest(k, n, CHANCE, alternative="two-sided")

        res = bootstrap(
            (vals,), np.mean,
            n_resamples=2000,
            confidence_level=0.95,
            method="percentile",
        )

        results.append({
            "model":   m,
            "rate":    rate,
            "ci_low":  res.confidence_interval.low,
            "ci_high": res.confidence_interval.high,
            "p":       test.pvalue,
            "n":       n,
            "k":       k,
        })

        print(
            f"  {MODEL_LABELS[m]:<25} "
            f"rate={rate:.1%}  k={k}/{n}  p={test.pvalue:.4f}  "
            f"{_sig_marker(test.pvalue, rate)}"
        )

    # ── Part B: permutation test on between-model variance ───────────
    obs_rates        = np.array([r["rate"] for r in results])
    obs_variance     = obs_rates.var()
    model_labels_arr = df["model"].values
    outcome_arr      = df["pop_over_acc"].values
    perm_vars        = np.empty(N_PERM)

    rng = np.random.default_rng(42)
    for i in range(N_PERM):
        shuffled     = rng.permutation(model_labels_arr)
        perm_rates   = np.array([
            outcome_arr[shuffled == m].mean() for m in MODEL_ORDER
        ])
        perm_vars[i] = perm_rates.var()

    p_perm = (perm_vars >= obs_variance).mean()

    print(f"\n  Observed variance across models: {obs_variance:.6f}")
    print(f"  Permutation p (heterogeneity):   {p_perm:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel A — two-sided binomial ──────────────────────────────────
    ax      = axes[0]
    y       = np.arange(len(MODEL_ORDER))
    rates   = np.array([r["rate"]   for r in results])
    ci_low  = np.array([r["ci_low"] for r in results])
    ci_high = np.array([r["ci_high"] for r in results])
    pvals   = np.array([r["p"]      for r in results])

    ax.barh(
        y, rates,
        xerr=[rates - ci_low, ci_high - rates],
        color=MODEL_COLORS, alpha=0.85, height=0.55,
        error_kw={"elinewidth": 1.5, "capsize": 4, "ecolor": "#333"},
    )
    ax.axvline(CHANCE, linestyle="--", color="black", linewidth=1.2,
               label="Chance baseline (25%)")

    for i, (r, ch, p) in enumerate(zip(rates, ci_high - rates, pvals)):
        marker = _sig_marker(p, r)
        ax.text(r + ch + 0.015, i, f"{r:.1%} {marker}",
                va="center", fontsize=9, color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=10)
    ax.set_xlabel("Rate of popularity-over-accuracy recommendations", fontsize=10)
    ax.set_xlim(0, 0.85)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=9, frameon=False)
    ax.set_title(
        "A — Per-model bias rate",
        fontsize=10, fontweight="bold", loc="left",
    )

    # ── Panel B — permutation null distribution ───────────────────────
    ax = axes[1]
    ax.hist(perm_vars, bins=60, color="#aac4e0", edgecolor="white",
            linewidth=0.4, label="Permutation null")
    ax.axvline(obs_variance, color="#d6604d", linewidth=2.2,
               label=f"Observed variance\n(p={p_perm:.4f})")
    ax.set_xlabel("Variance in per-model bias rates", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_xscale("log")

    ax.legend(fontsize=9, frameon=False)
    ax.set_title(
        "B — Between-model heterogeneity",
        fontsize=10, fontweight="bold", loc="left",
    )

    _save("fig3_bias_significance.png")


# ── Fig 4 — Popularity signals vs accuracy scatter ────────────────────

def fig4_popularity_vs_accuracy(df):
    """
    2×4 scatter grid showing Spearman ρ between each popularity signal
    and pipeline accuracy, split by prompt type (rows).

    Columns: citations, GitHub stars, GitHub forks, combined pop_index.
    Weak ρ across all signals confirms that popularity and accuracy are
    largely independent dimensions — the models are not simply recommending
    the best tools, they are recommending the most popular ones.
    """
    metrics = [
        ("pipeline_citations", "Citations"),
        ("pipeline_stars",     "GitHub Stars"),
        ("pipeline_forks",     "GitHub Forks"),
        ("pop_index",          "Combined index"),
    ]
    prompts = ["expert", "naive"]
    results = []

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)

    for row, prompt in enumerate(prompts):
        sub = df[df["prompt_type"] == prompt]

        for col, (metric, label) in enumerate(metrics):
            ax   = axes[row, col]
            x    = sub[metric].values
            y    = sub["pipeline_accuracy"].values
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

            if len(x) < 2 or np.std(x) == 0:
                continue

            color = ACC_COLOR if prompt == "expert" else POP_COLOR
            ax.scatter(x, y, alpha=0.5, s=30, edgecolors="white",
                       linewidths=0.3, color=color)

            coef   = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, coef[0] * x_line + coef[1],
                    color="black", linewidth=1.5)

            rho, p = spearmanr(x, y)
            results.append({"prompt": prompt, "metric": metric,
                             "label": label, "rho": rho, "p": p})

            p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"  # type: ignore
            ax.set_title(f"{label}\nρ={rho:.2f}, {p_str}", fontsize=9)

            if row == 1:
                ax.set_xlabel(label, fontsize=9)
            if col == 0:
                ax.set_ylabel(
                    f"{prompt.capitalize()} prompt\nPipeline accuracy",
                    fontsize=9,
                )

    print("\n[Fig 4] Spearman correlations:")
    print(pd.DataFrame(results).to_string(index=False))

    fig.suptitle(
        "Figure 4 — Popularity signals vs pipeline accuracy\n"
        "(weak or absent correlation supports a popularity bias)",
        fontsize=11, fontweight="bold", x=0.01, ha="left",
    )

    _save("fig4_popularity_vs_accuracy.png")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(FIGURES_PATH, exist_ok=True)

    df = load_data()

    print(
        f"Dataset: {len(df)} rows | "
        f"{df['model'].nunique()} models | "
        f"{df['task'].nunique()} tasks | "
        f"{df['prompt_type'].nunique()} prompt types\n"
    )

    print("Generating main figures...")

    fig1_recommendation_distribution(df)
    fig2_bias_heatmap_combined(df)
    fig3_bias_significance(df)
    fig4_popularity_vs_accuracy(df)

    print("\nDone — 4 main figures written to:", FIGURES_PATH)