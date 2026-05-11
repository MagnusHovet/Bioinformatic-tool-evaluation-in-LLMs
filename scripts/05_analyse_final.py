"""
05_analyse_final.py
===================
Main publication figures for:
  "Popularity vs accuracy in AI-generated bioinformatic pipeline recommendations"

Figures
-------
  Fig 1 — Popularity-bias rate per model (95 % bootstrap CI)
  Fig 2 — Popularity signals vs pipeline accuracy (Spearman scatter)
  Fig 3 — Bias-rate heatmap by model × task
  Fig 4 — Conflict test: P(high accuracy) as a function of conflict strength

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
from scipy.stats import spearmanr, bootstrap
import statsmodels.formula.api as smf

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

# ── Helpers ───────────────────────────────────────────────────────────

def _save(name):
    plt.tight_layout()
    path = os.path.join(FIGURES_PATH, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

# ── Data ──────────────────────────────────────────────────────────────

def load_data():
    """
    Load scored data and engineer all features needed for main figures.

    pop_index
        Stars and forks (r = 0.94) are averaged first into a single
        GitHub signal, then averaged with citations — giving each
        source 50 % weight rather than letting GitHub dominate at 2/3.
        All inputs are already normalised 0–1, so pop_index stays on
        the same scale as pipeline_accuracy.

    pop_over_acc
        Primary outcome variable.  Within each task, a pipeline is
        flagged (1) when it is in the top-popularity half AND the
        bottom-accuracy half.  Chance baseline = 25 %.

    conflict_strength
        pop_index − pipeline_accuracy.  Positive values mean the
        model recommended a pipeline more popular than accurate.

    high_accuracy
        Task-relative binary: 1 if a pipeline's accuracy is above
        the within-task median, 0 otherwise.  Used as the outcome in
        the Fig 4 logistic model.  Using within-task splits avoids
        the quasi-complete separation that arises when a global split
        is combined with a conflict_strength that is also task-driven.
    """

    df = pd.read_csv(SCORED_DATA_PATH)

    # Task-relative binary outcome for the logistic model in Fig 4
    df["high_accuracy"] = df.groupby("task")["pipeline_accuracy"].transform(
        lambda x: (x > x.median()).astype(int)
    )

    return df


# ── Fig 1 — Bias rate per model ───────────────────────────────────────

def fig1_bias_rate(df):
    """
    Horizontal bar chart of the popularity-over-accuracy rate per model,
    with 95 % bootstrap confidence intervals (2 000 resamples, percentile
    method).  Reference line at 25 % = chance baseline.
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

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.barh(
        y, rates,
        xerr=[ci_lows, ci_highs],
        color=MODEL_COLORS, alpha=0.85, height=0.55,
        error_kw={"elinewidth": 1.5, "capsize": 4, "ecolor": "#333333"},
    )
    ax.axvline(0.25, linestyle="--", color="black", linewidth=1.2,
               label="Expected by chance (25 %)")

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

    for i, (r, ch) in enumerate(zip(rates, ci_highs)):
        ax.text(r + ch + 0.015, i, f"{r:.1%}", va="center", fontsize=9, color="#333")

    _save("fig1_bias_rate.png")


# ── Fig 2 — Popularity vs accuracy scatter ────────────────────────────

def fig2_popularity_vs_accuracy(df):
    """
    2 × 4 scatter grid.  Columns: citations, stars, forks, combined
    index.  Rows: expert / naive prompt type.  Each subplot shows the
    Spearman ρ between that popularity metric and pipeline accuracy.
    Weak ρ confirms that popularity and accuracy are largely independent
    dimensions, giving the bias finding in Fig 1 its interpretive weight.
    """

    metrics = [
        ("pipeline_citations", "Citations"),
        ("pipeline_stars",     "GitHub Stars"),
        ("pipeline_forks",     "GitHub Forks"),
        ("pop_index",          "Combined index"),
    ]
    prompts = ["expert", "naive"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)

    for row, prompt in enumerate(prompts):
        sub = df[df["prompt_type"] == prompt]

        for col, (metric, label) in enumerate(metrics):
            ax  = axes[row, col]
            x   = sub[metric].values
            y   = sub["pipeline_accuracy"].values

            color = ACC_COLOR if prompt == "expert" else POP_COLOR
            ax.scatter(x, y, alpha=0.5, s=30, edgecolors="white",
                       linewidths=0.3, color=color)

            coef   = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, coef[0] * x_line + coef[1],
                    color="black", linewidth=1.5)

            rho, p = spearmanr(x, y)
            p_str  = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
            ax.set_title(f"{label}\nρ={rho:.2f}, {p_str}", fontsize=9)

            if row == 1:
                ax.set_xlabel(label, fontsize=9)
            if col == 0:
                ax.set_ylabel(
                    f"{prompt.capitalize()} prompt\nPipeline accuracy",
                    fontsize=9,
                )

    fig.suptitle(
        "Figure 2 — Popularity signals vs pipeline accuracy\n"
        "(weak or absent correlation supports a popularity bias)",
        fontsize=11, fontweight="bold", x=0.01, ha="left",
    )

    _save("fig2_popularity_vs_accuracy.png")


# ── Fig 3 — Bias heatmap ──────────────────────────────────────────────

def fig3_bias_heatmap(df):
    """
    Heatmap of pop_over_acc rate for every model × task cell.
    Red = frequently chose popular over accurate; green = accuracy-aligned.
    Unpacks the aggregated rates in Fig 1 to reveal which tasks drive
    each model's overall bias.
    """

    pivot = (
        df.groupby(["model", "task"])["pop_over_acc"]
        .mean()
        .unstack()
    )
    pivot = pivot.loc[MODEL_ORDER, TASK_ORDER]
    pivot.index   = [MODEL_LABELS[m] for m in MODEL_ORDER]
    pivot.columns = [TASK_LABELS[t]  for t in TASK_ORDER]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

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
    cbar.set_label("Popularity-over-accuracy\nrecommendation rate", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )
    ax.set_title(
        "Figure 3 — Popularity bias rate by model and task\n"
        "(red = model frequently chose popular over accurate pipelines)",
        fontsize=11, fontweight="bold", loc="left",
    )

    _save("fig3_bias_heatmap.png")


# ── Fig 4 — Conflict test ─────────────────────────────────────────────

def _bootstrap_conflict_slopes(df, n_boot=500, seed=42):
    """
    Parametric bootstrap of the logistic regression slope for
    conflict_strength → high_accuracy.

    Only converged fits are retained to prevent runaway coefficients
    from inflating the CI.  With task-relative high_accuracy and the
    corrected pop_index there is no quasi-complete separation, so
    convergence failures should be rare (<1 %).

    Returns an array of slope estimates.
    """
    np.random.seed(seed)
    slopes = []

    for _ in range(n_boot):
        sample = df.sample(len(df), replace=True)
        try:
            m = smf.logit(
                "high_accuracy ~ conflict_strength",
                data=sample,
            ).fit(disp=0)
            if m.converged:
                slopes.append(m.params["conflict_strength"])
        except Exception:
            continue

    return np.array(slopes)


def fig4_conflict_test(df):
    """
    Logistic regression of P(high accuracy | conflict_strength).

    conflict_strength = pop_index − pipeline_accuracy.
    Positive conflict means the model chose a pipeline whose popularity
    exceeds its accuracy — the bias scenario.

    high_accuracy is defined within-task (above the task median) to
    avoid quasi-complete separation: a global accuracy split combined
    with a task-driven conflict_strength creates near-perfect
    predictability that causes coefficient explosion (~−500) and
    meaningless CIs.  The within-task split preserves the question
    ("did the model pick the more accurate pipeline for this task?")
    while keeping the regression well-conditioned.

    A negative slope means: the more the recommended pipeline's
    popularity exceeds its accuracy, the less likely the model was to
    have chosen a high-accuracy pipeline — direct evidence of bias.

    Per-model curves show whether this effect is uniform or driven by
    specific models.  Only models whose per-model fit converges are
    plotted; GPT-5.4 (7 unique conflict values) may be omitted.
    """

    df = df.copy()

    # ── Global model ──────────────────────────────────────────────────
    global_model = smf.logit(
        "high_accuracy ~ conflict_strength", data=df
    ).fit(disp=0)

    print("\n[Fig 4] Global model:")
    print(f"  conflict_strength coef = {global_model.params['conflict_strength']:.3f}")
    print(f"  p = {global_model.pvalues['conflict_strength']:.2e}")
    print(f"  converged = {global_model.converged}")

    # ── Bootstrap CI ──────────────────────────────────────────────────
    boot = _bootstrap_conflict_slopes(df, n_boot=500)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    print(f"  Bootstrap 95% CI: [{ci_low:.3f}, {ci_high:.3f}]  (n={len(boot)} converged)")

    # ── Plot ──────────────────────────────────────────────────────────
    x_range = np.linspace(
        df["conflict_strength"].min(),
        df["conflict_strength"].max(),
        200,
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    # Global curve
    y_global = global_model.predict(
        pd.DataFrame({"conflict_strength": x_range})
    )
    ax.plot(x_range, y_global, color="black", linewidth=2.5,
            label=f"Global  (β={global_model.params['conflict_strength']:.1f}, "
                  f"95% CI [{ci_low:.1f}, {ci_high:.1f}])")

    # Per-model curves — only plot converged fits
    for m, color in zip(MODEL_ORDER, MODEL_COLORS):
        sub = df[df["model"] == m]
        if len(sub) < 20:
            continue
        try:
            mod = smf.logit(
                "high_accuracy ~ conflict_strength", data=sub
            ).fit(disp=0)
            if not mod.converged:
                print(f"  Skipping {MODEL_LABELS[m]} — did not converge")
                continue
            y_mod = mod.predict(pd.DataFrame({"conflict_strength": x_range}))
            ax.plot(x_range, y_mod, linestyle="--", linewidth=1.4,
                    alpha=0.75, color=color, label=MODEL_LABELS[m])
        except Exception:
            continue

    ax.axhline(0.5, linestyle=":", color="#999", linewidth=1,
               label="P = 0.5 (no preference)")
    ax.axvline(0, linestyle=":", color="#999", linewidth=1)

    ax.set_xlabel("Conflict strength (popularity index − pipeline accuracy)", fontsize=10)
    ax.set_ylabel("P(pipeline is high-accuracy | conflict strength)", fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(
        "Figure 4 — Probability of selecting a high-accuracy pipeline\n"
        "as a function of popularity–accuracy conflict",
        fontsize=11, fontweight="bold", loc="left",
    )

    _save("fig4_conflict_test.png")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(FIGURES_PATH, exist_ok=True)

    df = load_data()

    print(f"Dataset: {len(df)} rows | "
          f"{df['model'].nunique()} models | "
          f"{df['task'].nunique()} tasks | "
          f"{df['prompt_type'].nunique()} prompt types\n")

    print("Generating main figures...")

    fig1_bias_rate(df)
    fig2_popularity_vs_accuracy(df)
    fig3_bias_heatmap(df)
    fig4_conflict_test(df)

    print("\nDone — 4 main figures written to:", FIGURES_PATH)