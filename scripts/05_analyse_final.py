"""
05_analyse_final.py
===================
Main publication figures for:
  "Popularity vs accuracy in AI-generated bioinformatic pipeline recommendations"

Figures
-------
  Fig M1 — Methods figure: tool benchmark overlay with quadrant annotation
  Fig 1  — Distribution of all four recommendation outcome categories
            (A = by model, B = by task)
  Fig 2  — Bias-rate heatmap by model × task
            (A–D = one panel per binary outcome)
  Fig 3  — Statistical evidence for popularity bias and between-model
            heterogeneity (A = binomial Pop>Acc, B = binomial Acc>Pop)
  Fig 4  — Popularity signals vs pipeline accuracy (2×2 scatter,
            naive/expert overlaid as two colours per panel)
  Fig 5  — Effect of prompt type on pipeline accuracy (paired plot)

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
import matplotlib.patches as mpatches
from scipy.stats import spearmanr, bootstrap, binomtest, wilcoxon

from config.settings import SCORED_DATA_PATH, FIGURES_PATH, MODELS, TASKS, BENCHMARK_FINAL_PATH


# ── Global font sizes (uncomfortably large, then a bit bigger) ────────

FONT_BASE   = 22   # axis labels, tick labels
FONT_TITLE  = 18   # panel and figure titles
FONT_ANNOT  = 20   # in-cell annotations, bar labels
FONT_LEGEND = 25   # legend text
FONT_TOOL   = 14

plt.rcParams.update({
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "font.family":        "sans-serif",
    "font.size":          FONT_BASE,
    "axes.titlesize":     FONT_TITLE,
    "axes.labelsize":     FONT_BASE,
    "xtick.labelsize":    FONT_BASE,
    "ytick.labelsize":    FONT_BASE,
    "legend.fontsize":    FONT_LEGEND,
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

# ── Quadrant colour scheme ────────────────────────────────────────────
# Order: hi acc + hi pop, hi acc + lo pop, lo acc + lo pop, lo acc + hi pop
QUAD_COLORS = {
    "Hi Acc + Hi Pop": "#3D8DD9",   # hi acc + hi pop — blue
    "Hi Acc + Lo Pop": "#2918C1",   # hi acc + lo pop — dark blue
    "Lo Acc + Lo Pop": "#76767D",   # lo acc + lo pop — grey
    "Lo Acc + Hi Pop": "#C05020",   # lo acc + hi pop — orange-red
}

# Ordered list for stacked bars
QUAD_ORDER = ["Hi Acc + Lo Pop","Hi Acc + Hi Pop","Lo Acc + Lo Pop", "Lo Acc + Hi Pop"]

# Column names corresponding to each quadrant label
QUAD_TO_COL = {
    "Hi Acc + Hi Pop": "acc_and_pop",
    "Hi Acc + Lo Pop": "acc_over_pop",
    "Lo Acc + Lo Pop": "no_acc_and_pop",
    "Lo Acc + Hi Pop": "pop_over_acc",
}

MODEL_COLORS = ["#2166ac", "#4dac26", "#b8860b", "#d6604d", "#c51b7d"]

ACC_COLOR    = QUAD_COLORS["Hi Acc + Hi Pop"]
POP_COLOR    = QUAD_COLORS["Hi Acc + Lo Pop"]
EXPERT_COLOR = QUAD_COLORS["Hi Acc + Hi Pop"]
NAIVE_COLOR  = QUAD_COLORS["Lo Acc + Lo Pop"]

CHANCE = 0.25
RNG    = np.random.default_rng(42)


# ── Helpers ───────────────────────────────────────────────────────────

def _save(name):
    """Save the current figure to FIGURES_PATH and close it."""
    plt.tight_layout()
    path = os.path.join(FIGURES_PATH, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

def _sig_marker(p, rate, chance=CHANCE, show_p=True):
    """
    Return significance marker encoding significance level, direction,
    and optionally exact p-value.

    Above chance: ↑  Below chance: ↓  Not significant: ns
    Asterisks: * p<0.05  ** p<0.01  *** p<0.001
    """

    # format p-value once
    p_str = f"p={p:.3g}"

    if p >= 0.05:
        marker = "ns"
        direction = ""
    else:
        direction = "↑" if rate > chance else "↓"

        if p < 0.001:
            marker = f"{direction}***"
        elif p < 0.01:
            marker = f"{direction}**"
        else:
            marker = f"{direction}*"

    if show_p:
        return f"{marker} ({p_str})"
    else:
        return marker


def load_data():
    """Load scored pipeline data from SCORED_DATA_PATH."""
    return pd.read_csv(SCORED_DATA_PATH)


def order_by_acc_over_pop(df, group_by, quartile, ascend):
    """Return group values sorted by mean of quartile column."""
    return (
        df.groupby(group_by)[quartile]
        .mean()
        .sort_values(ascending=ascend)
        .index.tolist()
    )


def figM1_tool_benchmark_overlay():
    """
    Methods figure: scatter of all benchmark tools coloured by quadrant.

    """
    from adjustText import adjust_text

    df = pd.read_csv(BENCHMARK_FINAL_PATH)
    df["tool"] = df["tool"].str.lower().str.strip()

    def rank_norm(s):
        ranks = s.rank(method="average", ascending=False, na_option="keep")
        n = s.notna().sum()
        if n <= 1:
            return s - s
        return 1 - (ranks - 1) / (n - 1)

    df["citations_norm"] = rank_norm(df["citations_2026"])
    df["stars_norm"]     = rank_norm(df["github_stars"])
    df["forks_norm"]     = rank_norm(df["github_forks"])

    github_mean     = df[["stars_norm", "forks_norm"]].mean(axis=1)
    df["pop_index"] = np.where(
        github_mean.notna(),
        (df["citations_norm"] + github_mean) / 2,
        df["citations_norm"],
    )

    df = df[df["accuracy"].notna() & df["pop_index"].notna()].copy()

    med_acc = df["accuracy"].median()
    med_pop = df["pop_index"].median()

    def quadrant(row):
        hi_acc = row["accuracy"]  > med_acc
        hi_pop = row["pop_index"] > med_pop
        if   hi_acc and     hi_pop: return "Hi Acc + Hi Pop"
        elif hi_acc and not hi_pop: return "Hi Acc + Lo Pop"
        elif not hi_acc and not hi_pop: return "Lo Acc + Lo Pop"
        else:                       return "Lo Acc + Hi Pop"

    df["quadrant"] = df.apply(quadrant, axis=1)

    fig, ax = plt.subplots(figsize=(14, 8))

    # ── Quadrant shading ──────────────────────────────────────────────
 
    # ── Scatter points ────────────────────────────────────────────────
    all_texts = []
    for quad in QUAD_ORDER:
        color = QUAD_COLORS[quad]
        sub   = df[df["quadrant"] == quad]
        ax.scatter(
            sub["pop_index"], sub["accuracy"],
            color=color, s=80, alpha=0.9,
            linewidths=0.5, edgecolors="white",
            zorder=3, label=quad,
        )
        for _, row in sub.iterrows():
            all_texts.append(ax.text(
                row["pop_index"], row["accuracy"],
                row["tool"], fontsize=FONT_TOOL, alpha=0.85,
            ))

    adjust_text(
        all_texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="grey", lw=0.4),
        expand_points=(2.0, 2.0),
        force_text=(1.2, 1.2),
    )

    # ── Median split lines ────────────────────────────────────────────
    ax.axvline(med_pop, linestyle="--", linewidth=1.2,
               color="#555", alpha=0.6, zorder=2)
    ax.axhline(med_acc, linestyle="--", linewidth=1.2,
               color="#555", alpha=0.6, zorder=2, label="Median split")

    # ── Diagonal line from min to max ────────────────────────────────
    x_diag = np.array([0.0, 1.0])
    y_diag = np.array([1.0, 0.0])   # identity line: acc = pop
    ax.plot(x_diag, y_diag, color="black", linewidth=1.8,
            linestyle="--", alpha=0.5, zorder=2, label="acc = pop")


    # ── Quadrant labels directly on figure ───────────────────────────
    quad_label_positions = {
        "Hi Acc + Hi Pop": (0.78, 1.05),
        "Hi Acc + Lo Pop": (0.18, 1.05),
        "Lo Acc + Lo Pop": (0.18, 0.08),
        "Lo Acc + Hi Pop": (0.78, 0.08),
    }
    for quad, (fx, fy) in quad_label_positions.items():
        ax.text(
            fx, fy, quad,
            transform=ax.transAxes,
            fontsize=FONT_BASE, fontweight="bold",
            color=QUAD_COLORS[quad],
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=QUAD_COLORS[quad], alpha=0.7, lw=1.5),
        )

    ax.set_xlabel("Popularity index", fontsize=FONT_BASE)
    ax.set_ylabel("Accuracy (benchmark)", fontsize=FONT_BASE)
    ax.set_xlim(-0.02, 1.15)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=FONT_LEGEND - 6, frameon=False,
              loc="upper center",
              bbox_to_anchor=(0.5, -0.10),
              ncol=3,
              handlelength=1.5)

    _save("figM1_tool_benchmark_overlay.png")


# ── Fig 1 — Recommendation outcome distribution (A=model, B=task) ─────

def fig1_recommendation_distribution(df):
    """
    Two-panel stacked horizontal bar chart showing all four binary
    outcome proportions
    """
    cols   = [QUAD_TO_COL[q] for q in QUAD_ORDER]
    labels = QUAD_ORDER
    colors = [QUAD_COLORS[q] for q in QUAD_ORDER]

    sorted_models = order_by_acc_over_pop(df, "model", "acc_over_pop", True)
    sorted_tasks  = order_by_acc_over_pop(df, "task",  "acc_over_pop", True)

    panels = [
        ("model", sorted_models, [MODEL_LABELS[m] for m in sorted_models], "A — Grouped by model"),
        ("task",  sorted_tasks,  [TASK_LABELS[t]  for t in sorted_tasks],  "B — Grouped by task"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(32, 12))

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
                        fontsize=FONT_ANNOT+7, color="white", fontweight="bold",
                    )
            bottoms += vals

        ax.set_xlabel("Proportion of recommendations (%)", fontsize=FONT_BASE+4)
        ax.set_xlim(0, 100)
        ax.set_title(panel_title, fontsize=FONT_TITLE+6, fontweight="bold", loc="left")
        ax.tick_params(axis="both", labelsize=FONT_BASE+10)

    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors],
        labels=labels,
        fontsize=FONT_LEGEND+8,
        frameon=False,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.08),
        handlelength=1.4,
    )

    _save("fig1_recommendation_distribution.png")


# ── Fig 2 — Bias heatmap (A–D, one per binary outcome) ───────────────

def fig2_bias_heatmap_combined(df):
    """
    2×2 grid of heatmaps in professor's requested quadrant order:
      Panel A — hi acc + hi pop  (acc_and_pop)
      Panel B — hi acc + lo pop  (acc_over_pop)
      Panel C — lo acc + lo pop  (no_acc_and_pop)
      Panel D — lo acc + hi pop  (pop_over_acc)

    Y-axis labels on left column only, X-axis on bottom row only.
    """
    outcomes = [
        ("acc_and_pop",    "A — Hi Acc + Hi Pop"),
        ("acc_over_pop",   "B — Hi acc + Lo Pop"),
        ("no_acc_and_pop", "C — Lo acc + Lo Pop"),
        ("pop_over_acc",   "D — Lo acc + Hi Pop"),
    ]

    sorted_models = order_by_acc_over_pop(df, "model", "acc_over_pop", True)
    sorted_tasks  = order_by_acc_over_pop(df, "task",  "acc_over_pop", True)

    fig, axes = plt.subplots(2, 2, figsize=(22, 15))

    for idx, (ax, (outcome, panel_title)) in enumerate(
        zip(axes.flatten(), outcomes)
    ):
        row = idx // 2
        col = idx  % 2

        pivot = (
            df.groupby(["model", "task"])[outcome]
            .mean()
            .unstack()
        )
        pivot         = pivot.loc[sorted_models[::-1], sorted_tasks[::-1]]
        pivot.index   = [MODEL_LABELS[m] for m in sorted_models[::-1]]
        pivot.columns = [TASK_LABELS[t]  for t in sorted_tasks[::-1]]

        ax.imshow(pivot.values, cmap="viridis", vmin=0, vmax=0.7,
                  aspect="auto")

        if row == 1:
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, fontsize=FONT_BASE)
        else:
            ax.set_xticks([])

        if col == 0:
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=FONT_BASE)
        else:
            ax.set_yticks([])

        ax.set_title(panel_title, fontsize=FONT_TITLE+6, fontweight="bold",
                     loc="left")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val   = pivot.values[i, j]
                color = "black" if val > 0.45 else "white"
                ax.text(j, i, f"{val:.0%}",
                        ha="center", va="center",
                        fontsize=FONT_ANNOT+4, fontweight="bold", color=color)


    _save("fig2_bias_heatmap_combined.png")


# ── Fig 3 — Binomial test + permutation test ──────────────────────────

def fig3_bias_significance(df):
    """
    Two-panel significance figure with exact p-values.

    Panel A — Pop > Acc binomial test per model.
    Panel B — Acc > Pop binomial test per model.
    Both two-sided against 25% chance baseline.
    Exact p-values shown in annotation labels.
    """
    N_PERM = 10_000

    def run_binomial(df, outcome):
        results = []
        for m in MODEL_ORDER:
            vals = df[df["model"] == m][outcome].values
            n    = len(vals)
            k    = int(vals.sum())
            rate = k / n
            test = binomtest(k, n, CHANCE, alternative="two-sided")
            res  = bootstrap(
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
                f"rate={rate:.1%}  k={k}/{n}  "
                f"{_sig_marker(test.pvalue, rate)}"
            )
        return results

    print("\n[Fig 3A] Pop > Acc:")
    results_pop = run_binomial(df, "pop_over_acc")

    print("\n[Fig 3B] Acc > Pop:")
    results_acc = run_binomial(df, "acc_over_pop")

    # ── Permutation test ──────────────────────────────────────────────
    obs_rates        = np.array([r["rate"] for r in results_pop])
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
    p_perm_str = f"{p_perm:.4f}" if p_perm >= 0.0001 else f"{p_perm:.2e}"
    print(f"\n  Observed variance: {obs_variance:.6f}")
    print(f"  Permutation p:     {p_perm_str}")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(28, 11))

    sorted_models_pop = order_by_acc_over_pop(df, "model", "pop_over_acc", True)
    sorted_models_acc = order_by_acc_over_pop(df, "model", "acc_over_pop", True)

    def reorder_results(results, order):
        rd = {r["model"]: r for r in results}
        return [rd[m] for m in order]

    panels = [
        (axes[0], reorder_results(results_pop, sorted_models_pop),
         sorted_models_pop,
         "A — Lo Acc + Hi Pop",
         "Rate of pop-over-accuracy recommendations"),
        (axes[1], reorder_results(results_acc, sorted_models_acc),
         sorted_models_acc,
         "B — Hi Acc + Lo pop",
         "Rate of acc-over-popularity recommendations"),
    ]

    for ax, results, sorted_models, title, xlabel in panels:
        y       = np.arange(len(sorted_models))
        rates   = np.array([r["rate"]   for r in results])
        ci_low  = np.array([r["ci_low"] for r in results])
        ci_high = np.array([r["ci_high"] for r in results])
        pvals   = np.array([r["p"]      for r in results])
        colors = [MODEL_COLORS[MODEL_ORDER.index(m)] for m in sorted_models]

        ax.barh(
            y, np.maximum(rates, 0.005),
            xerr=[rates - ci_low, ci_high - rates],
            color=colors, alpha=0.80, height=0.55,
            error_kw={"elinewidth": 2.0, "capsize": 5, "ecolor": "#333"},
        )
        ax.axvline(CHANCE, linestyle="--", color="black", linewidth=1.5)

        for i, (r, ch, p) in enumerate(zip(rates, ci_high - rates, pvals)):
            marker = _sig_marker(p, r)
            ax.text(r + ch + 0.015, i, f"{r:.1%} {marker}",
                    va="center", fontsize=FONT_ANNOT, color="#333")

        ax.set_yticks(y)
        ax.set_yticklabels(
            [MODEL_LABELS[m] for m in sorted_models], fontsize=FONT_BASE
        )
        ax.set_xlabel(xlabel, fontsize=FONT_BASE)
        ax.set_xlim(0, 0.95)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0%}")
        )
        ax.legend(fontsize=FONT_LEGEND, frameon=False)
        ax.set_title(title, fontsize=FONT_TITLE+5, fontweight="bold", loc="left")
        ax.tick_params(axis="both", labelsize=FONT_BASE)


    _save("fig3_bias_significance.png")


# ── Fig 4 — Popularity signals vs accuracy (2×2, naive/expert overlaid) ─

def fig4_popularity_vs_accuracy(df):
    """
    2×2 scatter grid showing Spearman ρ between each popularity signal
    and pipeline accuracy.  Naive and expert overlaid in each panel.
    Exact p-values reported.  Per-prompt and combined ρ both shown.
    """
    metrics = [
        ("pipeline_citations", "Citations",      "A"),
        ("pipeline_stars",     "GitHub Stars",   "B"),
        ("pipeline_forks",     "GitHub Forks",   "C"),
        ("pop_index",          "Combined index", "D"),
    ]

    prompt_style = {
        "expert": (EXPERT_COLOR, "Expert"),
        "naive":  (NAIVE_COLOR,  "Naive"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    print("\n[Fig 4] Spearman correlations:")
    print(f"  {'Metric':<20} {'Prompt':<10} {'ρ':>7}  {'p':>10}")
    print("  " + "-" * 54)

    for idx, (metric, label, panel_letter) in enumerate(metrics):
        ax = axes[idx // 2][idx % 2]

        for prompt, (color, prompt_label) in prompt_style.items():
            sub  = df[df["prompt_type"] == prompt]
            x    = sub[metric].values
            y    = sub["pipeline_accuracy"].values
            mask = np.isfinite(x) & np.isfinite(y)
            x_m, y_m = x[mask], y[mask]

            if len(x_m) < 2:
                continue

            ax.scatter(x_m, y_m, alpha=0.45, s=35,
                       edgecolors="white", linewidths=0.3,
                       color=color, label=prompt_label, zorder=3)


            rho, p = spearmanr(x_m, y_m)
            p_str  = f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}"
            print(f"  {label:<20} {prompt_label:<10} {rho:>+7.3f}  p={p_str}")

        # Combined Spearman
        x_all    = df[metric].values
        y_all    = df["pipeline_accuracy"].values
        mask_all = np.isfinite(x_all) & np.isfinite(y_all)
        rho_all, p_all = spearmanr(x_all[mask_all], y_all[mask_all])
        p_all_str = f"{p_all:.4f}" if p_all >= 0.0001 else f"{p_all:.2e}"
        print(f"  {label:<20} {'combined':<10} {rho_all:>+7.3f}  p={p_all_str}")
        print("  " + "-" * 54)

        # Per-prompt rho for title
        rhos = {}
        for prompt in ["expert", "naive"]:
            sub  = df[df["prompt_type"] == prompt]
            x    = sub[metric].values
            y    = sub["pipeline_accuracy"].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() > 1:
                rhos[prompt], _ = spearmanr(x[mask], y[mask])
            else:
                rhos[prompt] = float("nan")

        ax.set_title(
            f"{panel_letter}  {label}\n"
            f"ρ ={rho_all:.2f} (p={p_all_str})",
            fontsize=FONT_BASE, fontweight="bold", loc="left",
        )

        ax.set_xlabel(f"{label} (rank-normalised)", fontsize=FONT_BASE+4)
        ax.set_ylabel("Pipeline accuracy", fontsize=FONT_BASE+4)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.35, 1.05)
        ax.tick_params(axis="both", labelsize=FONT_BASE+4)

        # Median reference lines
        ax.axvline(np.median(x_all[mask_all]), color="grey",
                   linewidth=0.8, linestyle="--", alpha=0.4)
        ax.axhline(np.median(y_all[mask_all]), color="grey",
                   linewidth=0.8, linestyle="--", alpha=0.4)

        if idx == 0:
            ax.legend(fontsize=FONT_LEGEND, frameon=False,
                      title="Prompt type", title_fontsize=FONT_LEGEND)

    _save("fig4_popularity_vs_accuracy.png")


# ── Fig 5 — Prompt effect on pipeline accuracy ────────────────────────

def fig5_prompt_effect(df):
    """
    Two-panel paired strip plot: pipeline accuracy and popularity index
    under naive vs expert prompts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=False)

    metrics = [
        ("pipeline_accuracy", "A  Pipeline accuracy",
         "Task-averaged pipeline accuracy", 0.45, 0.95),
        ("pop_index", "B  Popularity index",
         "Task-averaged popularity index", 0.40, 0.90),
    ]

    TASK_MARKERS = {
        "genome_assembly": "o",
        "variant_detection": "s",
        "RNA-seq": "^",
        "Phylogeny": "D",
    }

    prompt_colors = {"naive": NAIVE_COLOR, "expert": EXPERT_COLOR}

    for ax, (metric, panel_title, ylabel, ymin, ymax) in zip(axes, metrics):

        # ── task-level aggregation (for stats only) ──────────────────
        task_means = (
            df.groupby(["task", "prompt_type"])[metric]
            .mean()
            .reset_index()
        )

        pivot = task_means.pivot(
            index="task", columns="prompt_type", values=metric
        ).dropna()

        stat, p = wilcoxon(pivot["naive"], pivot["expert"])
        p_str = f"p={p:.4f}" if p >= 0.0001 else f"p={p:.2e}"

        print(f"\n[Fig 5] {metric} Wilcoxon: stat={stat:.3f}, {p_str}")

        # ── CI + points per task × prompt (UPDATED) ────────────────
        summary = (
            df.groupby(["task", "prompt_type"])[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        summary["se"] = summary["std"] / np.sqrt(summary["count"])
        summary["ci"] = 1.96 * summary["se"]

        for _, row in summary.iterrows():
            task = row["task"]
            prompt = row["prompt_type"]

            x_pos = 1 if prompt == "naive" else 2
            jitter = RNG.uniform(-0.05, 0.05)

            marker = TASK_MARKERS.get(task, "o")

            ax.errorbar(
                x_pos + jitter,
                row["mean"],
                yerr=row["ci"],
                fmt="none",
                ecolor=prompt_colors[prompt],
                elinewidth=1.8,
                capsize=4,
                alpha=0.8,
                zorder=2,
            )

            ax.scatter(
                x_pos + jitter,
                row["mean"],
                alpha=0.9,
                s=120,
                color=prompt_colors[prompt],
                marker=marker,
                zorder=3,
            )

        # ── Connecting lines (paired tasks) ─────────────────────────
        for _, row in pivot.iterrows():
            ax.plot(
                [1, 2],
                [row["naive"], row["expert"]],
                color="grey",
                alpha=0.35,
                linewidth=1.2,
                zorder=1,
            )

        # ── Grand mean line ─────────────────────────────────────────
        ax.plot(
            [1, 2],
            [pivot["naive"].mean(), pivot["expert"].mean()],
            marker="D",
            markersize=11,
            linewidth=2.8,
            color="black",
            zorder=4,
            label=(
                f"Grand mean "
                f"({pivot['naive'].mean():.3f} → "
                f"{pivot['expert'].mean():.3f})"
            ),
        )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Naive", "Expert"], fontsize=FONT_BASE + 6)
        ax.set_ylabel(ylabel, fontsize=FONT_BASE+5)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0.6, 2.4)
        ax.tick_params(axis="both", labelsize=FONT_BASE)

        ax.set_title(
            f"{panel_title}\nWilcoxon {p_str}, n={len(pivot)} tasks",
            fontsize=FONT_TITLE+6,
            fontweight="bold",
            loc="left",
        )

        ax.legend(fontsize=FONT_LEGEND, frameon=False)

    # ── Shared legends ─────────────────────────────────────────────
    prompt_handles = [
        mpatches.Patch(color=NAIVE_COLOR, label="Naive"),
        mpatches.Patch(color=EXPERT_COLOR, label="Expert"),
    ]

    task_handles = [
        plt.scatter([], [], marker=m, color="grey", s=100,
                    label=TASK_LABELS.get(t, t))
        for t, m in TASK_MARKERS.items()
    ]

    fig.legend(
        handles=prompt_handles + task_handles,
        fontsize=FONT_LEGEND,
        frameon=False,
        loc="lower center",
        ncol=len(prompt_handles) + len(task_handles),
        bbox_to_anchor=(0.5, -0.06),
    )

    _save("fig5_prompt_effect.png")

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

    figM1_tool_benchmark_overlay()
    fig1_recommendation_distribution(df)
    fig2_bias_heatmap_combined(df)
    fig3_bias_significance(df)
    fig4_popularity_vs_accuracy(df)
    fig5_prompt_effect(df)

    print("\nDone — figures written to:", FIGURES_PATH)