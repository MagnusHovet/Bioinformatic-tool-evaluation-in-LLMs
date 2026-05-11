import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import statsmodels.api as sm

from config.settings import SCORED_DATA_PATH, FIGURES_PATH

# ── Style ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

def _save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, name), bbox_inches="tight")
    plt.close()
    print(f"Saved: {name}")

# ── Data ──────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(SCORED_DATA_PATH)

    df = df.dropna(subset=[
        "pipeline_accuracy",
        "pipeline_citations",
        "pipeline_stars",
        "pipeline_forks"
    ])

    # ── unified popularity index ──
    scaler = StandardScaler()

    df[["cit_z", "stars_z", "forks_z"]] = scaler.fit_transform(
        df[["pipeline_citations", "pipeline_stars", "pipeline_forks"]]
    )

    df["popularity_index"] = df[["cit_z", "stars_z", "forks_z"]].mean(axis=1)

    # binary outcome for model
    df["high_accuracy"] = (
        df["pipeline_accuracy"] > df["pipeline_accuracy"].median()
    ).astype(int)

    return df

# ──────────────────────────────────────────────────────────────────────
# FIGURE 1 — POPULARITY ≠ ACCURACY
# ──────────────────────────────────────────────────────────────────────

def fig1_popularity_vs_accuracy(df):

    metrics = [
        ("pipeline_citations", "Citations"),
        ("pipeline_stars", "GitHub stars"),
        ("pipeline_forks", "GitHub forks"),
        ("popularity_index", "Combined popularity"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)

    for ax, (col, label) in zip(axes, metrics):

        x = df[col].values 
        y = df["pipeline_accuracy"].values

        ax.scatter(x, y, alpha=0.6, s=50, edgecolors="white", linewidths=0.4)

        if len(df) > 5:
            coef = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = coef[0] * x_line + coef[1]
            ax.plot(x_line, y_line, color="black", linewidth=2)

            rho, p = spearmanr(x, y)
            ax.set_title(f"{label}\nρ={rho:.2f}, p={p:.3f}")
        else:
            ax.set_title(label)

        ax.set_xlabel(label)

    axes[0].set_ylabel("Pipeline accuracy")
    ax.set_ylim(0.45, 1.0)

    fig.suptitle(
        "Popularity vs accuracy",
        fontsize=12
    )

    _save("fig1_popularity_vs_accuracy.png")

# ──────────────────────────────────────────────────────────────────────
# FIGURE 2 — NO ROBUST PREDICTIVE EFFECT
# ──────────────────────────────────────────────────────────────────────

def fig2_predictive_model(df):

    model = smf.logit(
        "high_accuracy ~ cit_z + stars_z + forks_z + popularity_index",
        data=df
    ).fit()

    params = model.params
    conf = model.conf_int()
    conf.columns = ["low", "high"]

    coef_df = pd.concat([params, conf], axis=1)
    coef_df.columns = ["coef", "low", "high"]
    coef_df = coef_df.drop("Intercept")

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.errorbar(
        coef_df["coef"],
        coef_df.index,
        xerr=[
            coef_df["coef"] - coef_df["low"],
            coef_df["high"] - coef_df["coef"]
        ],
        fmt="o"
    )

    ax.axvline(0, linestyle="--", color="black")

    ax.set_title("Figure 2 — No robust predictive effect of popularity signals")
    ax.set_xlabel("Logistic regression coefficient (high accuracy)")

    _save("fig2_predictive_effect.png")

# ──────────────────────────────────────────────────────────────────────
# BOOTSTRAP FUNCTION
# ──────────────────────────────────────────────────────────────────────

def bootstrap_slopes(df, n_boot=500, seed=42):

    np.random.seed(seed)

    slopes = []

    for _ in range(n_boot):

        sample = df.sample(len(df), replace=True)

        try:
            m = smf.logit(
                "high_accuracy ~ conflict_strength",
                data=sample
            ).fit(disp=0)

            slopes.append(m.params["conflict_strength"])

        except:
            continue

    return np.array(slopes)


# ──────────────────────────────────────────────────────────────────────
# FIGURE 3 — IMPROVED CONFLICT ANALYSIS
# ──────────────────────────────────────────────────────────────────────

def fig3_conflict_test(df):

    df = df.copy()
    df["conflict_strength"] = df["popularity_index"] - df["pipeline_accuracy"]

    # ── 1. GLOBAL MODEL ────────────────────────────────────────────────
    global_model = smf.logit(
        "high_accuracy ~ conflict_strength",
        data=df
    ).fit()

    print("\nGLOBAL MODEL")
    print(global_model.summary())

    # bootstrap CI
    boot = bootstrap_slopes(df)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    print(f"\nBootstrap slope CI: [{ci_low:.3f}, {ci_high:.3f}]")

    # ── 2. INTERACTION MODEL (MODEL-SPECIFIC EFFECTS) ────────────────
    if "model" in df.columns:

        interaction_model = smf.logit(
            "high_accuracy ~ conflict_strength * C(model)",
            data=df
        ).fit()

        print("\nINTERACTION MODEL")
        print(interaction_model.summary())

    # ── 3. PLOT: SPLIT BY MODEL ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.linspace(
        df["conflict_strength"].min(),
        df["conflict_strength"].max(),
        200
    )

    # global prediction
    y_global = global_model.predict(
        pd.DataFrame({"conflict_strength": x})
    )

    ax.plot(x, y_global, label="Global", linewidth=3)

    # per-model curves
    if "model" in df.columns:

        for m in df["model"].unique():

            sub = df[df["model"] == m]

            if len(sub) < 20:
                continue

            try:
                mod = smf.logit(
                    "high_accuracy ~ conflict_strength",
                    data=sub
                ).fit(disp=0)

                y = mod.predict(pd.DataFrame({"conflict_strength": x}))

                ax.plot(x, y, linestyle="--", alpha=0.7, label=str(m))

            except:
                continue

    ax.axhline(0.5, linestyle="--", color="black", linewidth=1)

    ax.set_title("Figure 3 — Bias under popularity–accuracy conflict")
    ax.set_xlabel("Conflict strength (popularity − accuracy)")
    ax.set_ylabel("P(high accuracy selected)")
    ax.legend(fontsize=8)

    _save("fig3_conflict_test.png")

# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(FIGURES_PATH, exist_ok=True)

    df = load_data()

    print("Generating 3-figure publication pipeline...\n")

    fig1_popularity_vs_accuracy(df)
    fig2_predictive_model(df)
    fig3_conflict_test(df)

    print("\nDone.")