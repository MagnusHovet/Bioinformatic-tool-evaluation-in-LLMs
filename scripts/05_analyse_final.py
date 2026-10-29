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
  Fig 2  — Paper Fig 2: outcome distribution per model; the Lo Acc + Hi Pop
            rate is tested against random choice (printed for the caption)
  Fig 3  — Paper Fig 3: per-model outcome bars stacked by task (A–E)
            and the naive → expert prompt effect (F)
  Fig 3s — Statistical evidence for popularity bias and between-model
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

from typing import get_args

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr, bootstrap, binomtest, wilcoxon

from config.settings import SCORED_DATA_PATH, NULL_PATH, FIGURES_PATH, MODELS, TASKS, BENCHMARK_FINAL_PATH, TASK_TO_MODEL


# ── Global font sizes (uncomfortably large, then a bit bigger) ────────

FONT_BASE   = 22   # axis labels, tick labels
FONT_TITLE  = 18   # panel and figure titles
FONT_ANNOT  = 20   # in-cell annotations, bar labels
FONT_LEGEND = 25   # legend text

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
    # 2026-10 rerun
    "anthropic/claude-sonnet-5-5":   "Claude Sonnet 5.5",
    "deepseek/deepseek-v4-pro":      "DeepSeek V4 Pro",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "openai/gpt-6.1-sol":            "GPT-6.1 Sol",
    "openai/gpt-6-luna":             "GPT-6 Luna",
}

TASK_LABELS = {
    "Phylogeny":         "Phylogeny",
    "RNA-seq":           "RNA-seq",
    "genome_assembly":   "Genome assembly",
    "variant_detection": "Variant detection",
}

# Colour + marker shape per problem type, used to distinguish tasks in scatter figures
# (Paul Tol "muted" subset — colour-blind safe, distinct from the quadrant colours)
TASK_COLORS = {
    "genome_assembly":   "#332288",
    "variant_detection": "#DDCC77",
    "RNA-seq":           "#44AA99",
    "Phylogeny":         "#AA4499",
}
TASK_MARKERS = {
    "genome_assembly":   "o",
    "variant_detection": "s",
    "RNA-seq":           "^",
    "Phylogeny":         "D",
}

# ── Quadrant colour scheme ────────────────────────────────────────────
# Ordinal, worst → best (Okabe–Ito derived, colour-blind safe)
QUAD_COLORS = {
    "Lo Acc + Lo Pop": "#A6A6A6",   # grey
    "Lo Acc + Hi Pop": "#D55E00",   # vermillion — the popularity-bias outcome
    "Hi Acc + Lo Pop": "#56B4E9",   # sky blue
    "Hi Acc + Hi Pop": "#0072B2",   # blue
}
# Ink for in-segment labels, chosen per fill for contrast
QUAD_TEXT = {
    "Lo Acc + Lo Pop": "#1a1a1a",
    "Lo Acc + Hi Pop": "white",
    "Hi Acc + Lo Pop": "#1a1a1a",
    "Hi Acc + Hi Pop": "white",
}

# Ordered list for stacked bars: worst → best
QUAD_ORDER = ["Lo Acc + Lo Pop", "Lo Acc + Hi Pop", "Hi Acc + Lo Pop", "Hi Acc + Hi Pop"]
QUAD_SHORT = ["Lo Acc\nLo Pop", "Lo Acc\nHi Pop", "Hi Acc\nLo Pop", "Hi Acc\nHi Pop"]

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

# Print-size typography for the paper figures (Fig 2, Fig 3): figures are
# drawn close to their printed size, so these are near real point sizes.
PRINT_RC = {
    "font.size":       12,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.edgecolor":  "#444",
    "xtick.color":     "#444",
    "ytick.color":     "#444",
}


# ── Helpers ───────────────────────────────────────────────────────────

def _save(name, formats=("png",), dpi="figure"):
    """Save the current figure to FIGURES_PATH (one file per format) and close it."""
    plt.tight_layout()
    stem = os.path.splitext(name)[0]
    for ext in formats:
        path = os.path.join(FIGURES_PATH, f"{stem}.{ext}")
        plt.savefig(path, bbox_inches="tight", dpi=dpi)
        print(f"  Saved → {path}")
    plt.close()


def _holm(pvals):
    """Holm–Bonferroni adjusted p-values, returned in input order."""
    p      = np.asarray(pvals, dtype=float)
    adj    = np.empty_like(p)
    m      = len(p)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adj[i]  = min(running, 1.0)
    return adj


def _fmt_p(p):
    """Format a p-value for figure text, e.g. 'P = 0.34' or 'P = 5.7×10⁻⁹'."""
    if p >= 0.001:
        return f"P = {p:.2f}" if p >= 0.01 else f"P = {p:.3f}"
    mant, exp = f"{p:.1e}".split("e")
    return rf"P = {mant}$\times$10$^{{{int(exp)}}}$"

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


def load_null():
    """Load the per-task random-choice null (share of candidate space per outcome)."""
    return pd.read_csv(NULL_PATH).set_index("task")


def _null_rate(df, null, model, outcome):
    """
    Expected rate of an outcome for one model under random tool choice:
    the per-task null rates weighted by that model's number of
    recommendations in each task.
    """
    n_per_task = df.loc[df["model"] == model, "task"].value_counts()
    return (null.loc[n_per_task.index, outcome] * n_per_task).sum() / n_per_task.sum()


def order_by_acc_over_pop(df, group_by, quartile, ascend):
    """Return group values sorted by mean of quartile column."""
    return (
        df.groupby(group_by)[quartile]
        .mean()
        .sort_values(ascending=ascend)
        .index.tolist()
    )


def _tool_task_map():
    """
    Map each benchmark tool name to the task(s) whose pipeline schema
    (config.settings.TASK_TO_MODEL) references it as a valid choice.

    A tool referenced by more than one task's schema (e.g. an aligner
    used in both the RNA-seq and variant-detection pipelines) maps to
    all of them.
    """
    mapping = {}
    for task, model_cls in TASK_TO_MODEL.items():
        for field in model_cls.model_fields.values():
            for tool in get_args(field.annotation):
                mapping.setdefault(tool, []).append(task)
    return mapping


def _load_tool_benchmark_by_task():
    """
    Load the benchmark tool table, compute the popularity index, and
    expand to one row per (tool, task) pair using the tool→task mapping
    derived from the pipeline schemas. A tool referenced by more than
    one task's schema (e.g. an aligner shared between RNA-seq and
    variant detection) appears once per task it actually belongs to.
    """
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

    tool_task_map = _tool_task_map()
    df["task"]    = df["tool"].map(tool_task_map)
    df            = df.explode("task")
    df            = df[df["task"].notna()].copy()

    return df


def figM1_tool_benchmark_overlay():
    """
    Methods figure: scatter of all benchmark tools, coloured and shaped
    by problem type (task), with per-task median-split lines marking
    each task's own hi/lo accuracy × popularity quadrants.
    """
    df = _load_tool_benchmark_by_task()

    fig, ax = plt.subplots(figsize=(10, 10))

    # ── Scatter points, coloured + shaped by problem type ─────────────
    for task in TASK_ORDER:
        sub = df[df["task"] == task]
        if sub.empty:
            continue
        ax.scatter(
            sub["pop_index"], sub["accuracy"],
            color=TASK_COLORS[task], marker=TASK_MARKERS[task],
            s=90, alpha=0.85,
            linewidths=0.5, edgecolors="white",
            zorder=3, label=TASK_LABELS[task],
        )

        # ── Per-task median-split lines ("quarters" for this problem) ──
        med_acc = sub["accuracy"].median()
        med_pop = sub["pop_index"].median()
        ax.axvline(med_pop, linestyle="--", linewidth=1.2,
                   color=TASK_COLORS[task], alpha=0.45, zorder=2)
        ax.axhline(med_acc, linestyle="--", linewidth=1.2,
                   color=TASK_COLORS[task], alpha=0.45, zorder=2)

    # ── Diagonal reference line ────────────────────────────────────────
    x_diag = np.array([0.0, 1.0])
    y_diag = np.array([1.0, 0.0])   # identity line: acc = pop
    ax.plot(x_diag, y_diag, color="black", linewidth=1.5,
            linestyle="--", alpha=0.4, zorder=2, label="acc = pop")

    ax.set_xlabel("Popularity index", fontsize=FONT_BASE)
    ax.set_ylabel("Accuracy (benchmark)", fontsize=FONT_BASE)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_box_aspect(1)
    ax.legend(fontsize=FONT_LEGEND - 8, frameon=False,
              loc="upper center",
              bbox_to_anchor=(0.5, -0.12),
              ncol=3,
              handlelength=1.5)

    _save("figM1_tool_benchmark_overlay.png")


def figM1_tool_benchmark_faceted():
    """
    Methods figure (small-multiples variant): one scatter panel per
    problem type, each with its own median-split crosshair confined to
    that panel. Avoids the overlapping-crosshair clutter of the single
    combined-overlay version while keeping the per-task quadrant split
    just as legible.
    """
    df = _load_tool_benchmark_by_task()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    for task, ax in zip(TASK_ORDER, axes.flat):
        sub   = df[df["task"] == task]
        color = TASK_COLORS[task]

        ax.scatter(
            sub["pop_index"], sub["accuracy"],
            color=color, marker=TASK_MARKERS[task],
            s=80, alpha=0.85,
            linewidths=0.5, edgecolors="white",
            zorder=3,
        )

        # ── Median-split lines ("quarters" for this problem) ──────────
        ax.axvline(sub["pop_index"].median(), linestyle="--", linewidth=1.2,
                   color=color, alpha=0.55, zorder=2)
        ax.axhline(sub["accuracy"].median(), linestyle="--", linewidth=1.2,
                   color=color, alpha=0.55, zorder=2)

        # ── Diagonal reference line ─────────────────────────────────────
        ax.plot([0.0, 1.0], [1.0, 0.0], color="black", linewidth=1.2,
                linestyle="--", alpha=0.3, zorder=1)

        ax.set_title(TASK_LABELS[task], fontsize=FONT_TITLE, fontweight="bold",
                     color=color, loc="left")
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.set_box_aspect(1)
        ax.tick_params(axis="both", labelsize=FONT_BASE - 4)

    fig.supxlabel("Popularity index", fontsize=FONT_BASE)
    fig.supylabel("Accuracy (benchmark)", fontsize=FONT_BASE)

    _save("figM1_tool_benchmark_faceted.png")


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


# ── Fig 2 (paper) — outcome distribution + Hi Acc + Lo Pop test column ─

def fig2_bias_summary(df, null):
    """
    Paper Figure 2. One stacked horizontal bar per model showing the four
    outcome categories ordered worst → best (Lo/Lo, Lo Acc + Hi Pop,
    Hi Acc + Lo Pop, Hi/Hi). Models are ordered by their Lo Acc + Hi Pop
    rate (the primary outcome), highest at the top. That rate, its exact
    (Clopper–Pearson) 95% CI and a two-sided binomial test against the
    random-choice expectation (Holm-adjusted across models) are printed
    for the figure caption rather than drawn on the figure.
    """
    order = order_by_acc_over_pop(df, "model", "pop_over_acc", True)   # bottom → top
    cols  = [QUAD_TO_COL[q] for q in QUAD_ORDER]
    means = df.groupby("model")[cols].mean().mul(100).reindex(order)

    stats = []
    for m in order:
        vals = df.loc[df["model"] == m, "pop_over_acc"]
        p0   = _null_rate(df, null, m, "pop_over_acc")
        test = binomtest(int(vals.sum()), len(vals), p0, alternative="two-sided")
        ci   = test.proportion_ci(confidence_level=0.95, method="exact")
        stats.append((vals.mean(), ci.low, ci.high, test.pvalue, p0))
    p_adj = _holm([s[3] for s in stats])

    print("\n[Fig 2] Lo Acc + Hi Pop vs random choice (exact CI, Holm-adjusted):")
    for m, (rate, lo, hi, p, p0), pa in zip(order[::-1], stats[::-1], p_adj[::-1]):
        print(f"  {MODEL_LABELS[m]:<20} {rate:6.1%}  [{lo:.1%}, {hi:.1%}]  "
              f"null={p0:.1%}  p={p:.3g}  p_holm={pa:.3g}")
    print("  Category shares (%):")
    print(df.groupby("model")[cols].mean().mul(100).round(1)
            .rename(index=MODEL_LABELS, columns=dict(zip(cols, QUAD_ORDER))).to_string())

    with plt.rc_context(PRINT_RC):
        fig, ax = plt.subplots(figsize=(6.4, 3.3))
        y    = np.arange(len(order))
        left = np.zeros(len(order))

        for q, col in zip(QUAD_ORDER, cols):
            vals = means[col].values
            ax.barh(y, vals, left=left, height=0.7, color=QUAD_COLORS[q],
                    edgecolor="white", linewidth=1.5, label=q)
            for i, (v, l) in enumerate(zip(vals, left)):
                if v >= 8:
                    ax.text(l + v / 2, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=10.5, color=QUAD_TEXT[q])
            left += vals

        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_LABELS[m] for m in order])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, len(order) - 0.5)
        ax.set_xlabel("Share of recommendations (%)")

        ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.2),
                  bbox_transform=ax.transAxes, ncol=4, frameon=False,
                  handlelength=1.0, columnspacing=1.0, fontsize=10)

        _save("fig2_bias_summary.png", formats=("png", "pdf"), dpi=300)


# ── Fig 3 (paper) — per-model outcomes by task (A–E) + prompt effect (F) ─

def fig3_model_task_prompt(df, null):
    """
    Paper Figure 3, 2×3 grid.

    Panels A–E — one per model (same order as Fig 2, top to bottom).
      x: the four outcome categories ordered worst → best; bar height is
      the share of that model's recommendations, stacked by task so each
      segment is that task's contribution. A rising profile left → right
      indicates a model that favours accurate tools. Dashed marks show
      the expected share under random tool choice.

    Panel F — prompt effect. Task-mean popularity (x) vs accuracy (y),
      with an arrow per task from the naive to the expert prompt and a
      black arrow for the mean across tasks. Changes are tested with a
      Wilcoxon signed-rank test over model × task pairs (n = 20), since
      a test over n = 4 tasks cannot reach P < 0.125.
    """
    models = order_by_acc_over_pop(df, "model", "pop_over_acc", False)
    cols   = [QUAD_TO_COL[q] for q in QUAD_ORDER]
    x      = np.arange(len(QUAD_ORDER))
    ymax   = 108

    # ── Prompt-effect statistics (model × task pairs) ─────────────────
    mt = (df.groupby(["model", "task", "prompt_type"])[["pipeline_accuracy", "pop_index"]]
            .mean().unstack("prompt_type"))
    prompt_stats = {}
    for metric in ["pipeline_accuracy", "pop_index"]:
        naive, expert = mt[metric]["naive"], mt[metric]["expert"]
        _, p = wilcoxon(naive, expert)
        prompt_stats[metric] = (expert.mean() - naive.mean(), p, len(naive))
        print(f"\n[Fig 3F] {metric}: naive {naive.mean():.3f} → expert {expert.mean():.3f}, "
              f"Wilcoxon p={p:.3g} (n={len(naive)} model×task pairs)")

    with plt.rc_context(PRINT_RC):
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2))

        # ── A–E: per-model category bars stacked by task ─────────────
        for k, (ax, m) in enumerate(zip(axes.flat, models)):
            sub    = df[df["model"] == m]
            n      = len(sub)
            bottom = np.zeros(len(QUAD_ORDER))

            for task in TASK_ORDER:
                share = sub.loc[sub["task"] == task, cols].sum().values / n * 100
                ax.bar(x, share, bottom=bottom, width=0.72,
                       color=TASK_COLORS[task], edgecolor="white", linewidth=1.2,
                       label=TASK_LABELS[task], zorder=2)
                bottom += share

            # Random-choice expectation per category, drawn across each bar
            expected = [_null_rate(df, null, m, c) * 100 for c in cols]
            ax.hlines(expected, x - 0.42, x + 0.42, linestyles="--",
                      colors="#222", linewidth=1.3, zorder=3)

            for xi, tot, ex in zip(x, bottom, expected):
                ax.text(xi, max(tot, ex) + 1.5, f"{tot:.0f}%", ha="center",
                        va="bottom", fontsize=11, color="#333")

            ax.set_xticks(x)
            ax.set_xticklabels(QUAD_SHORT, fontsize=10)
            ax.tick_params(axis="x", length=0)
            ax.set_ylim(0, ymax)
            ax.set_yticks(range(0, 101, 20))
            if k % 3 == 0:
                ax.set_ylabel("Share of recommendations (%)")
            else:
                ax.set_yticklabels([])
            ax.set_title(f"{'ABCDE'[k]}  {MODEL_LABELS[m]}",
                         loc="left", fontweight="bold")

        # ── F: prompt effect as naive → expert arrows ─────────────────
        ax = axes.flat[5]
        tm = (df.groupby(["task", "prompt_type"])[["pipeline_accuracy", "pop_index"]]
                .mean().unstack("prompt_type"))

        def _arrow(p0, a0, p1, a1, color, lw):
            ax.annotate("", xy=(p1, a1), xytext=(p0, a0),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                        mutation_scale=16, shrinkA=0, shrinkB=0),
                        zorder=3)
            ax.scatter(p0, a0, s=55, facecolor="white", edgecolor=color,
                       linewidth=2, zorder=4)

        for task in TASK_ORDER:
            p0, p1 = tm.loc[task, ("pop_index", "naive")], tm.loc[task, ("pop_index", "expert")]
            a0, a1 = tm.loc[task, ("pipeline_accuracy", "naive")], tm.loc[task, ("pipeline_accuracy", "expert")]
            _arrow(p0, a0, p1, a1, TASK_COLORS[task], 2.2)

        means = tm.mean()
        _arrow(means[("pop_index", "naive")], means[("pipeline_accuracy", "naive")],
               means[("pop_index", "expert")], means[("pipeline_accuracy", "expert")],
               "black", 2.6)
        ax.text(means[("pop_index", "expert")] + 0.008, means[("pipeline_accuracy", "expert")],
                "Mean", fontsize=10.5, va="center", color="#1a1a1a")

        d_acc, p_acc, n_pairs = prompt_stats["pipeline_accuracy"]
        d_pop, p_pop, _       = prompt_stats["pop_index"]
        ax.text(0.98, 0.04,
                f"Accuracy {d_acc:+.3f}, {_fmt_p(p_acc)}\n"
                f"Popularity {d_pop:+.3f}, {_fmt_p(p_pop)}\n"
                f"Wilcoxon, n = {n_pairs} model–task pairs",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, color="#333", linespacing=1.35)

        ax.set_xlabel("Popularity index (task mean)")
        ax.set_ylabel("Pipeline accuracy (task mean)")
        ax.set_xlim(0.45, 0.80)
        ax.set_ylim(0.55, 0.88)
        ax.grid(color="#e6e6e6", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title("F  Naive → expert prompt", loc="left", fontweight="bold")
        ax.legend(handles=[
                      plt.Line2D([], [], marker="o", linestyle="none", markersize=7,
                                 markerfacecolor="white", markeredgecolor="#333",
                                 markeredgewidth=2, label="Naive"),
                      plt.Line2D([], [], marker=">", linestyle="none", markersize=7,
                                 color="#333", label="Expert"),
                  ],
                  loc="upper left", frameon=False, fontsize=10, handletextpad=0.3)

        # ── Shared legend: tasks + chance line ────────────────────────
        handles = [mpatches.Patch(color=TASK_COLORS[t], label=TASK_LABELS[t])
                   for t in TASK_ORDER]
        handles.append(plt.Line2D([], [], linestyle="--", color="#555",
                                  linewidth=1.3, label="Random-choice expectation"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   frameon=False, bbox_to_anchor=(0.5, -0.035))

        _save("fig3_model_task_prompt.png", formats=("png", "pdf"), dpi=300)


# ── Fig 3s — Binomial test + permutation test ──────────────────────────

def fig3_bias_significance(df, null):
    """
    Two-panel significance figure with exact p-values.

    Panel A — Pop > Acc binomial test per model.
    Panel B — Acc > Pop binomial test per model.
    Both two-sided against the random-choice expectation.
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
            p0   = _null_rate(df, null, m, outcome)
            test = binomtest(k, n, p0, alternative="two-sided")
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
                "p0":      p0,
            })
            print(
                f"  {MODEL_LABELS[m]:<25} "
                f"rate={rate:.1%}  k={k}/{n}  null={p0:.1%}  "
                f"{_sig_marker(test.pvalue, rate, chance=p0)}"
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
        p0s     = np.array([r["p0"]     for r in results])
        colors = [MODEL_COLORS[MODEL_ORDER.index(m)] for m in sorted_models]

        ax.barh(
            y, np.maximum(rates, 0.005),
            xerr=[rates - ci_low, ci_high - rates],
            color=colors, alpha=0.80, height=0.55,
            error_kw={"elinewidth": 2.0, "capsize": 5, "ecolor": "#333"},
        )
        ax.vlines(p0s, y - 0.4, y + 0.4, linestyle="--", color="black", linewidth=1.5)

        for i, (r, ch, p, p0) in enumerate(zip(rates, ci_high - rates, pvals, p0s)):
            marker = _sig_marker(p, r, chance=p0)
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

    df   = load_data()
    null = load_null()

    print(
        f"Dataset: {len(df)} rows | "
        f"{df['model'].nunique()} models | "
        f"{df['task'].nunique()} tasks | "
        f"{df['prompt_type'].nunique()} prompt types\n"
    )

    print("Generating main figures...")

    figM1_tool_benchmark_overlay()
    figM1_tool_benchmark_faceted()
    fig1_recommendation_distribution(df)
    fig2_bias_summary(df, null)
    fig3_model_task_prompt(df, null)
    fig3_bias_significance(df, null)
    fig4_popularity_vs_accuracy(df)
    fig5_prompt_effect(df)

    print("\nDone — figures written to:", FIGURES_PATH)