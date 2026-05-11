import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from scipy.stats import wilcoxon, spearmanr, gaussian_kde
from sklearn.preprocessing import StandardScaler

import statsmodels.formula.api as smf
import statsmodels.api as sm

from config.settings import SCORED_DATA_PATH, FIGURES_PATH

# ── Style ─────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "anthropic/claude-sonnet-4-6": "#1f77b4",
    "deepseek/deepseek-chat":       "#ff7f0e",
    "google/gemini-2.5-pro":        "#2ca02c",
    "openai/gpt-5.4":               "#d62728",
    "openai/gpt-5.4-mini":          "#9467bd",
}
FALLBACK_COLOR = "#7f7f7f"

TASK_MARKERS = {
    "Phylogeny": "o",
    "RNA-seq": "s",
    "genome_assembly": "^",
    "variant_detection": "D",
}
FALLBACK_MARKER = "x"

RNG = np.random.default_rng(42)

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

# ── Helpers ───────────────────────────────────────────────────────────

def _save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, name), bbox_inches="tight")
    plt.close()
    print(f"Saved: {name}")

# ── Load data ─────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(SCORED_DATA_PATH)

    df = df.dropna(subset=[
        "pipeline_accuracy",
        "pipeline_citations",
        "pipeline_stars",
        "pipeline_forks"
    ])

    # ── binary outcome ──
    median_acc = df["pipeline_accuracy"].median()
    df["high_accuracy"] = (df["pipeline_accuracy"] > median_acc).astype(int)

    df["delta"] = df["pipeline_citations"] - df["pipeline_accuracy"]

    # ─────────────────────────────────────────────
    # ⭐ POPULARITY INDEX (FIX FOR MULTICOLLINEARITY)
    # ─────────────────────────────────────────────

    scaler = StandardScaler()

    df[["cit_z", "stars_z", "forks_z"]] = scaler.fit_transform(
        df[["pipeline_citations", "pipeline_stars", "pipeline_forks"]]
    )

    df["popularity_index"] = df[["cit_z", "stars_z", "forks_z"]].mean(axis=1)

    return df

# ── Fig 1: Prompt effect ──────────────────────────────────────────────

def fig1_prompt_effect(df):

    # Aggregate to task-level means (same as before)
    task_means = (
        df.groupby(["task", "prompt_type"])["pipeline_accuracy"]
        .mean()
        .reset_index()
    )

    pivot = task_means.pivot(index="task", columns="prompt_type", values="pipeline_accuracy").dropna()

    # Wilcoxon (paired)
    if len(pivot) >= 3:
        stat, p = wilcoxon(pivot["naive"], pivot["expert"])
        p_str = f"p={p:.4f}"
        print(f"[Fig 1] Wilcoxon stat={stat:.3f}, p={p:.4f}")
    else:
        p_str = f"n={len(pivot)} tasks"

    fig, ax = plt.subplots(figsize=(5, 5))

    # Scatter points
    for x, prompt, color in zip([1, 2], ["naive", "expert"], ["#1f77b4", "#ff7f0e"]):
        vals = pivot[prompt]
        jitter = RNG.uniform(-0.05, 0.05, len(vals))

        ax.scatter(
            np.full(len(vals), x) + jitter,
            vals,
            alpha=0.7,
            s=80,
            color=color,
            label=f"{prompt.capitalize()} (mean={vals.mean():.3f})"
        )

    # Connect paired tasks
    for _, row in pivot.iterrows():
        ax.plot([1, 2], [row["naive"], row["expert"]],
                color="grey", alpha=0.3, linewidth=1)

    # Mean line
    ax.plot(
        [1, 2],
        [pivot["naive"].mean(), pivot["expert"].mean()],
        marker="D",
        linewidth=2.5,
        color="black",
        label=f"Mean ({pivot['naive'].mean():.3f} → {pivot['expert'].mean():.3f})"
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Naive prompt", "Expert prompt"])
    ax.set_ylabel("Task-averaged pipeline accuracy")
    ax.set_title(f"Effect of prompt type on accuracy\n({p_str}, n={len(pivot)} tasks)")
    ax.set_ylim(0.4, 1.05)
    ax.set_xlim(0.7, 2.3)

    ax.legend(fontsize=8)

    _save("fig1_prompt_effect.png")

# ── Fig 2: Delta distribution ─────────────────────────────────────────

def fig2_delta(df):

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x_grid = np.linspace(-0.8, 0.8, 500)

    colors = {
        "naive": "#888888",
        "expert": "#222222"
    }

    for prompt, linestyle in zip(["naive", "expert"], ["--", "-"]):

        vals = df[df["prompt_type"] == prompt]["delta"].dropna().values

        if len(vals) < 2:
            continue

        kde = gaussian_kde(vals, bw_method=0.35)
        density = kde(x_grid)

        ax.plot(
            x_grid,
            density,
            color=colors[prompt],
            linestyle=linestyle,
            linewidth=2.2,
            label=f"{prompt.capitalize()} (mean Δ={vals.mean():+.3f}, n={len(vals)})"
        )

        ax.fill_between(x_grid, density, alpha=0.08, color=colors[prompt])

    # Reference + bias regions
    ax.axvline(0, linestyle=":", color="black", linewidth=1.2, label="No bias (Δ=0)")

    ax.axvspan(0, 0.8, alpha=0.04, color="red", label="Popularity bias (Δ > 0)")
    ax.axvspan(-0.8, 0, alpha=0.04, color="blue", label="Accuracy-first (Δ < 0)")

    ax.set_xlabel("Δ = citations − accuracy")
    ax.set_ylabel("Density")

    ax.set_title(
        "Distribution of popularity–accuracy gap\n"
        "(dashed = naive, solid = expert)"
    )

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(bottom=0)

    ax.legend(fontsize=8)

    _save("fig2_delta.png")

# ── Fig 3: Accuracy vs delta ──────────────────────────────────────────

def fig3_acc_vs_delta_multi(df):

    # ─────────────────────────────────────────────
    # Create delta variants
    # ─────────────────────────────────────────────
    df = df.copy()

    df["delta_citations"] = df["pipeline_citations"] - df["pipeline_accuracy"]
    df["delta_stars"]     = df["pipeline_stars"]     - df["pipeline_accuracy"]
    df["delta_forks"]     = df["pipeline_forks"]     - df["pipeline_accuracy"]

    # Optional combined index (if not already present)
    if "popularity_index" not in df.columns:
        df["popularity_index"] = (
            df[["pipeline_citations", "pipeline_stars", "pipeline_forks"]]
            .mean(axis=1)
        )

    df["delta_combined"] = df["popularity_index"] - df["pipeline_accuracy"]

    metrics = [
        ("delta_citations", "Citations"),
        ("delta_stars", "GitHub stars"),
        ("delta_forks", "GitHub forks"),
        ("delta_combined", "Combined"),
    ]

    prompts = ["naive", "expert"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)

    for i, prompt in enumerate(prompts):
        for j, (col, label) in enumerate(metrics):

            ax = axes[i, j]

            sub = df[df["prompt_type"] == prompt]

            x = sub["pipeline_accuracy"].values
            y = sub[col].values

            # Scatter
            ax.scatter(
                x, y,
                alpha=0.6,
                s=45,
                edgecolors="white",
                linewidths=0.4
            )

            # Regression line
            if len(sub) > 5:
                coef = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = coef[0] * x_line + coef[1]

                ax.plot(x_line, y_line, color="black", linewidth=2)

                rho, p = spearmanr(x, y)

                ax.set_title(
                    f"{label}\nρ={rho:.2f}, p={p:.3f}",
                    fontsize=9
                )

            # Zero reference
            ax.axhline(0, linestyle="--", color="black", alpha=0.4)

            # Labels
            if i == 1:
                ax.set_xlabel("Pipeline accuracy")

            if j == 0:
                ax.set_ylabel(f"{prompt}\nΔ (popularity − accuracy)")

    fig.suptitle(
        "Accuracy vs popularity bias across metrics\n"
        "Negative slope → higher accuracy = less popularity alignment",
        fontsize=12
    )

    _save("fig3_acc_vs_delta_multi.png")

# ── Fig 4: Popularity vs accuracy ─────────────────────────────────────

def fig4_popularity_multi(df):

    metrics = [
        ("pipeline_citations", "Citations"),
        ("pipeline_stars", "GitHub stars"),
        ("pipeline_forks", "GitHub forks"),
        ("popularity_index", "Combined popularity"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)

    for ax, (col, label) in zip(axes, metrics):

        sub = df[[col, "pipeline_accuracy"]].dropna()

        x = sub[col].values
        y = sub["pipeline_accuracy"].values

        # Scatter
        ax.scatter(x, y, alpha=0.6, s=50, edgecolors="white", linewidths=0.4)

        # Regression line
        if len(sub) > 5:
            coef = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = coef[0] * x_line + coef[1]
            ax.plot(x_line, y_line, color="black", linewidth=2)

            # Spearman
            rho, p = spearmanr(x, y)
            ax.set_title(f"{label}\nρ={rho:.2f}, p={p:.3f}")
        else:
            ax.set_title(label)

        ax.set_xlabel(label)

    axes[0].set_ylabel("Pipeline accuracy")

    fig.suptitle(
        "Relationship between popularity signals and pipeline accuracy\n"
        "Comparing academic and community popularity metrics",
        fontsize=12
    )

    _save("fig4_popularity_multi.png")

# ── FIG 5: LOGISTIC REGRESSION (CORE MODEL) ──────────────────────────

def fig5_logistic(df):

    model = smf.logit(
        formula="""
        high_accuracy ~ popularity_index
        """,
        data=df
    ).fit()

    print(model.summary())

    fig, ax = plt.subplots()

    ax.bar(["popularity_index"], [model.params["popularity_index"]])
    ax.axhline(0, color="black")

    ax.set_title("Popularity index → high accuracy")

    _save("fig5_logistic.png")

# ── FIG 6: POPULARITY DECOMPOSITION (NEW KEY FIGURE) ─────────────────

def fig6_popularity_decomposition(df):

    model = smf.logit(
        formula="""
        high_accuracy ~ cit_z + stars_z + forks_z
        """,
        data=df
    ).fit()

    print(model.summary())

    coefs = model.params[1:]

    fig, ax = plt.subplots()

    ax.bar(coefs.index, coefs.values, color=["#999999", "#2ca02c", "#d62728"])

    ax.axhline(0, color="black")
    ax.set_title("Which popularity signals matter most?")

    _save("fig6_popularity_decomposition.png")

# ── MIXED EFFECTS (ROBUSTNESS) ────────────────────────────────────────

def fig6_mixed(df):

    model = smf.gee(
        formula="""
        high_accuracy ~ popularity_index
        """,
        groups="model",
        data=df,
        family=sm.families.Binomial()
    ).fit()

    print(model.summary()) 

# ── Fig 7: synthesis ──────────────────────────────────────────────────

def fig7(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics = [
        ("pipeline_citations", "Citations"),
        ("pipeline_stars", "Stars"),
        ("pipeline_forks", "Forks"),
    ]

    for ax, (col, name) in zip(axes, metrics):

        ax.scatter(df[col], df["pipeline_accuracy"])

        rho, _ = spearmanr(df[col], df["pipeline_accuracy"])
        ax.set_title(f"{name} (ρ={rho:.2f})")

    _save("fig7_synthesis.png")

# ── Fig 8: hypothesis test ──────────────────────────────────────────────────

def fig8_conflict_analysis(df):

    # ─────────────────────────────────────────────
    # Define thresholds (robust: median split)
    # ─────────────────────────────────────────────

    acc_thresh = df["pipeline_accuracy"].median()
    pop_thresh = df["popularity_index"].median()

    # Define groups
    df["conflict_group"] = "neutral"

    df.loc[
        (df["pipeline_accuracy"] < acc_thresh) &
        (df["popularity_index"] > pop_thresh),
        "conflict_group"
    ] = "popular_low_acc"

    df.loc[
        (df["pipeline_accuracy"] > acc_thresh) &
        (df["popularity_index"] < pop_thresh),
        "conflict_group"
    ] = "accurate_low_pop"

    # Keep only conflict cases
    sub = df[df["conflict_group"] != "neutral"].copy()

    # ─────────────────────────────────────────────
    # Compute probability of "high_accuracy"
    # ─────────────────────────────────────────────

    summary = (
        sub.groupby("conflict_group")["high_accuracy"]
        .mean()
        .reset_index()
    )

    print("\n[Fig 8] Conflict analysis:")
    print(summary)

    # ─────────────────────────────────────────────
    # Plot
    # ─────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(6, 4))

    order = ["popular_low_acc", "accurate_low_pop"]
    labels = [
        "High popularity\nLow accuracy",
        "High accuracy\nLow popularity"
    ]

    values = [
        summary.loc[summary["conflict_group"] == g, "high_accuracy"].values[0]
        if g in summary["conflict_group"].values else 0
        for g in order
    ]

    colors = ["#d62728", "#2ca02c"]

    ax.bar(labels, values, color=colors, alpha=0.8)

    ax.set_ylim(0, 1)
    ax.set_ylabel("P(high accuracy classification)")

    ax.set_title(
        "Do models favor popularity over accuracy\nwhen signals conflict?"
    )

    # Reference line
    ax.axhline(0.5, linestyle="--", color="black", alpha=0.5)

    _save("fig8_conflict_analysis.png")

# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(FIGURES_PATH, exist_ok=True)

    df = load_data()

    print("Generating figures...")

    fig1_prompt_effect(df)
    fig2_delta(df)
    fig3_acc_vs_delta_multi(df)
    fig4_popularity_multi(df)
    fig5_logistic(df)
    fig6_popularity_decomposition(df)
    fig6_mixed(df)
    fig7(df)
    fig8_conflict_analysis(df)

    print("Done.")