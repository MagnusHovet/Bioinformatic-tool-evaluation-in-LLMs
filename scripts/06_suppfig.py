"""
06_suppfig.py
=============
Supplementary figures for:
  "Popularity vs accuracy in AI-generated bioinformatic pipeline recommendations"

Figures
-------
  Supp 1 — Effect of prompt type on pipeline accuracy (paired plot)
  Supp 2 — Distribution of popularity–accuracy gap by prompt type (KDE)
  Supp 3 — Logistic regression: popularity signals as predictors of accuracy
  Supp 4 — Mean accuracy vs mean popularity index by model and prompt type

Usage
-----
  python 06_suppfig.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, gaussian_kde
import statsmodels.formula.api as smf

from config.settings import SCORED_DATA_PATH, FIGURES_PATH, MODELS, TASKS, ROLES


# ── Style ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
    "font.size":         10,
})

MODEL_ORDER  = MODELS
TASK_ORDER   = list(TASKS.keys())
PROMPT_TYPES = list(ROLES.keys())

MODEL_LABELS = {
    "anthropic/claude-sonnet-4-6": "Claude Sonnet 4.6",
    "deepseek/deepseek-chat":      "DeepSeek Chat",
    "google/gemini-2.5-pro":       "Gemini 2.5 Pro",
    "openai/gpt-5.4":              "GPT-5.4",
    "openai/gpt-5.4-mini":         "GPT-5.4 Mini",
}

NAIVE_COLOR  = "#888888"
EXPERT_COLOR = "#1a1a2e"
ACC_COLOR    = "#2166ac"
POP_COLOR    = "#d6604d"
MODEL_COLORS = ["#2166ac", "#4dac26", "#b8860b", "#d6604d", "#c51b7d"]

RNG = np.random.default_rng(42)


# ── Helpers ───────────────────────────────────────────────────────────

def _save(name):
    """Save the current figure to FIGURES_PATH and close it."""
    plt.tight_layout()
    path = os.path.join(FIGURES_PATH, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── Supp 1 — Prompt effect ────────────────────────────────────────────

def supp1_prompt_effect(df):
    """
    Paired strip plot of task-averaged accuracy under naive vs expert
    prompts.  Each point is one task; lines connect the same task across
    conditions.  Diamond markers show the grand mean.

    Wilcoxon signed-rank test (non-parametric, paired, n = 4 tasks).
    Note: with n = 4, power is very low — the test is indicative only.
    """
    task_means = (
        df.groupby(["task", "prompt_type"])["pipeline_accuracy"]
        .mean()
        .reset_index()
    )
    pivot = task_means.pivot(
        index="task", columns="prompt_type", values="pipeline_accuracy"
    ).dropna()

    stat, p = wilcoxon(pivot["naive"], pivot["expert"])
    print(f"[Supp 1] Wilcoxon: stat={stat:.3f}, p={p:.4f}")

    fig, ax = plt.subplots(figsize=(5, 5))

    prompt_style = {"naive": (1, NAIVE_COLOR), "expert": (2, EXPERT_COLOR)}

    for prompt, (x_pos, color) in prompt_style.items():
        vals   = pivot[prompt]
        jitter = RNG.uniform(-0.06, 0.06, len(vals))
        ax.scatter(
            np.full(len(vals), x_pos) + jitter, vals,
            alpha=0.75, s=90, color=color, zorder=3,
            label=f"{prompt.capitalize()} (mean={vals.mean():.3f})",
        )

    for _, row in pivot.iterrows():
        ax.plot([1, 2], [row["naive"], row["expert"]],
                color="grey", alpha=0.35, linewidth=1, zorder=2)

    ax.plot(
        [1, 2],
        [pivot["naive"].mean(), pivot["expert"].mean()],
        marker="D", markersize=9, linewidth=2.5, color="black", zorder=4,
        label=(
            f"Grand mean "
            f"({pivot['naive'].mean():.3f} → {pivot['expert'].mean():.3f})"
        ),
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Naive prompt", "Expert prompt"])
    ax.set_ylabel("Task-averaged pipeline accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.set_xlim(0.6, 2.4)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(
        "Supp. Figure 1 — Effect of prompt type on pipeline accuracy\n"
        f"Wilcoxon: p={p:.4f}, n={len(pivot)} tasks",
        fontsize=11, fontweight="bold", loc="left",
    )

    _save("supp1_prompt_effect.png")


# ── Supp 2 — Popularity–accuracy gap distribution ─────────────────────

def supp2_popularity_gap_distribution(df):
    """
    KDE density of the popularity–accuracy gap (pop_index − pipeline_accuracy)
    by prompt type.

    Δ > 0 means the model recommended a pipeline more popular than accurate.
    Both distributions sitting near Δ = 0 with a right tail shows that most
    recommendations are approximately calibrated, but some are strongly
    popularity-biased.  Expert prompting shifts the distribution toward zero
    if role-based prompting reduces the bias.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x_grid = np.linspace(-1.0, 1.0, 500)

    prompt_style = {
        "expert": (EXPERT_COLOR, "-"),
        "naive":  (NAIVE_COLOR,  "--"),
    }

    for prompt, (color, ls) in prompt_style.items():
        vals = df[df["prompt_type"] == prompt]["conflict_strength"].dropna().values
        kde  = gaussian_kde(vals, bw_method=0.2)

        ax.plot(x_grid, kde(x_grid),
                color=color, linestyle=ls, linewidth=2.2,
                label=f"{prompt.capitalize()} (mean Δ={vals.mean():+.3f})")
        ax.fill_between(x_grid, kde(x_grid), alpha=0.07, color=color)

    ax.axvline(0,    linestyle=":", color="black", linewidth=1.3, label="No bias (Δ=0)")
    ax.axvspan( 0,   1.0, alpha=0.04, color="red",  label="Popularity bias (Δ > 0)")
    ax.axvspan(-1.0, 0,   alpha=0.04, color="blue", label="Accuracy-first (Δ < 0)")

    ax.set_xlabel("Δ = popularity index − pipeline accuracy", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title(
        "Supp. Figure 2 — Distribution of popularity–accuracy gap by prompt type\n"
        "(right shift = models recommend pipelines more popular than optimal)",
        fontsize=11, fontweight="bold", loc="left",
    )

    _save("supp2_popularity_gap_distribution.png")


# ── Supp 3 — Popularity signals as predictors of accuracy ─────────────

def supp3_logistic_regression(df):
    """
    Logistic regression of high_accuracy on individual popularity signals.

    Two models are fitted separately to avoid collinearity (pop_index is
    the mean of the three components; including all four creates
    unidentifiable standard errors):

      Model A — individual components: citations, stars, forks
      Model B — combined pop_index

    Stars and forks may show opposite signs in Model A if they capture
    different facets of community adoption.  The combined index in Model B
    collapsing to near-zero would be consistent with popularity and accuracy
    being largely orthogonal dimensions.

    NOTE: a global median split is used for high_accuracy here (not the
    within-task split used in Fig 4) because we are asking whether any
    popularity signal predicts absolute accuracy, not task-relative ranking.
    """
    df = df.copy()

    # ── Global median split for this supplementary question ──────────
    df["high_accuracy"] = (
        df["pipeline_accuracy"] > df["pipeline_accuracy"].median()
    ).astype(int)

    model_A = smf.logit(
        "high_accuracy ~ pipeline_citations + pipeline_stars + pipeline_forks",
        data=df,
    ).fit(disp=0)

    model_B = smf.logit(
        "high_accuracy ~ pop_index",
        data=df,
    ).fit(disp=0)

    print("\n[Supp 3] Model A summary:")
    print(model_A.summary())
    print("\n[Supp 3] Model B summary:")
    print(model_B.summary())

    def _coef_df(model):
        p = model.params.drop("Intercept")
        c = model.conf_int().drop("Intercept")
        c.columns = ["low", "high"]
        return pd.concat([p, c], axis=1).rename(columns={0: "coef"})

    coef_df         = pd.concat([_coef_df(model_A), _coef_df(model_B)])
    coef_df.columns = ["coef", "low", "high"]

    label_map = {
        "pipeline_citations": "Citations  [Model A]",
        "pipeline_stars":     "GitHub Stars  [Model A]",
        "pipeline_forks":     "GitHub Forks  [Model A]",
        "pop_index":          "Combined index  [Model B]",
    }
    coef_df.index = [label_map.get(i, i) for i in coef_df.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos   = np.arange(len(coef_df))

    ax.errorbar(
        coef_df["coef"], y_pos,
        xerr=[
            coef_df["coef"] - coef_df["low"],
            coef_df["high"] - coef_df["coef"],
        ],
        fmt="o", color="#2166ac", markersize=7,
        elinewidth=1.8, capsize=5, ecolor="#555",
    )
    ax.axvline(0, linestyle="--", color="black", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_df.index, fontsize=9)
    ax.set_xlabel(
        "Logistic regression coefficient (log-odds of high accuracy)",
        fontsize=10,
    )
    ax.set_title(
        "Supp. Figure 3 — Popularity signals as predictors of pipeline accuracy\n"
        "(CIs crossing zero = no robust predictive effect)",
        fontsize=11, fontweight="bold", loc="left",
    )

    _save("supp3_logistic_regression.png")


# ── Supp 4 — Accuracy vs popularity by model and prompt type ─────────

def supp4_acc_vs_pop(df):
    """
    Paired bar chart: mean pipeline accuracy vs mean popularity index
    per model, split by prompt type (expert / naive).

    Arrows connect each model's accuracy bar to its popularity bar,
    making the direction and magnitude of the gap immediately visible.
    A model whose popularity bar is consistently higher than its
    accuracy bar is recommending pipelines more popular than they are
    accurate — the core bias signal shown here in absolute terms rather
    than as a binary rate.
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
               label="Pipeline accuracy", color=ACC_COLOR, alpha=0.85)
        ax.bar(x + width / 2, pop_vals, width,
               label="Popularity index",  color=POP_COLOR, alpha=0.85)

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
        "Supp. Figure 4 — Mean accuracy vs mean popularity of recommended pipelines\n"
        "by model and prompt type",
        fontsize=11, fontweight="bold", x=0.05, ha="left",
    )

    _save("supp4_acc_vs_pop.png")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(FIGURES_PATH, exist_ok=True)

    df = pd.read_csv(SCORED_DATA_PATH)

    print(
        f"Dataset: {len(df)} rows | "
        f"{df['model'].nunique()} models | "
        f"{df['task'].nunique()} tasks | "
        f"{df['prompt_type'].nunique()} prompt types\n"
    )

    print("Generating supplementary figures...")

    supp1_prompt_effect(df)
    supp2_popularity_gap_distribution(df)
    supp3_logistic_regression(df)
    supp4_acc_vs_pop(df)

    print("\nDone — 4 supplementary figures written to:", FIGURES_PATH)