"""
08_popularity_scope.py
======================
Sensitivity analysis for the scope of popularity rank-normalisation.

The primary analysis (04_score.py) rank-normalises popularity across all
benchmarked tools, while accuracy is normalised within each benchmark.
This script re-scores the recommendations with popularity normalised
within three scopes and reports how much the results change:

  global — across all benchmarked tools (primary analysis; reproduces
           04_score.py exactly)
  task   — across the candidate tools of each task
  step   — across the candidate tools of each pipeline step

Accuracy, thresholds (candidate-space medians), the random-choice null
and all tests are computed exactly as in the primary analysis.

Output
------
  runs/<RUN>/sensitivity/popularity_scope_shares.csv
  runs/<RUN>/sensitivity/popularity_scope_tests.csv

Usage
-----
  python 08_popularity_scope.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import importlib
import itertools
from typing import get_args

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from config.settings import RAW_DATA_PATH, BENCHMARK_FINAL_PATH, TASK_TO_MODEL, MODELS, RUN_DIR

score = importlib.import_module("04_score")

SCOPES   = ["global", "task", "step"]
OUTCOMES = {
    "no_acc_and_pop": "Lo Acc + Lo Pop",
    "pop_over_acc":   "Lo Acc + Hi Pop",
    "acc_over_pop":   "Hi Acc + Lo Pop",
    "acc_and_pop":    "Hi Acc + Hi Pop",
}
STEPS = {task: [get_args(f.annotation) for f in m.model_fields.values()]
         for task, m in TASK_TO_MODEL.items()}
OUT_DIR = os.path.join(RUN_DIR, "sensitivity")


# ── Scoring under a given normalisation scope ─────────────────────────

def tool_table(bench, scope):
    """
    Tool-level metrics keyed by (task, step, tool), with popularity
    rank-normalised within `scope`. Candidate pools count each tool once.
    """
    keys = pd.DataFrame([{"task": task, "step": s, "tool": t}
                         for task, steps in STEPS.items()
                         for s, opts in enumerate(steps) for t in opts])
    keys  = keys.merge(bench, on="tool", how="left")
    group = {"global": None, "task": ["task"], "step": ["task", "step"]}[scope]

    for raw, out in [("citations_2026", "cit"), ("github_stars", "stars"),
                     ("github_forks", "forks")]:
        if group is None:
            norm      = score.normalize_series(bench[raw])
            keys[out] = keys["tool"].map(dict(zip(bench["tool"], norm)))
        else:
            pools      = keys.drop_duplicates(group + ["tool"]).copy()
            pools[out] = pools.groupby(group)[raw].transform(score.normalize_series)
            keys       = keys.merge(pools[group + ["tool", out]],
                                    on=group + ["tool"], how="left")
    return keys.set_index(["task", "step", "tool"])


def pipeline_scores(df, tools):
    """Mean tool metrics per pipeline, then the composite popularity index."""
    rows = []
    for _, r in df.iterrows():
        idx = [(r["task"], s, t) for s, t in enumerate(r["tools"].split("|"))]
        sub = tools.reindex(idx)
        rows.append({**r.to_dict(),
                     "pipeline_accuracy":  sub["accuracy"].mean(),
                     "pipeline_citations": sub["cit"].mean(),
                     "pipeline_stars":     sub["stars"].mean(),
                     "pipeline_forks":     sub["forks"].mean()})
    return score.add_pop_index(pd.DataFrame(rows))


def score_scope(raw, bench, scope):
    """Classify recommendations and build the random-choice null for one scope."""
    tools = tool_table(bench, scope)
    rec   = pipeline_scores(raw, tools)
    space = pipeline_scores(pd.DataFrame(
        [{"task": t, "tools": "|".join(c)}
         for t, steps in STEPS.items() for c in itertools.product(*steps)]), tools)
    space = space[space["pipeline_accuracy"].notna()]

    thresholds = {"acc": space.groupby("task")["pipeline_accuracy"].median(),
                  "pop": space.groupby("task")["pop_index"].median()}
    rec   = score.classify(rec, thresholds)
    space = score.classify(space, thresholds)
    return rec, space.groupby("task")[list(OUTCOMES)].mean()


# ── Tests ─────────────────────────────────────────────────────────────

def holm(p):
    p, adj, running = np.asarray(p, float), np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i]  = min(running, 1.0)
    return adj


def binomial_tests(rec, null, scope):
    rows = []
    for outcome in ["pop_over_acc", "acc_over_pop"]:
        res = []
        for m in MODELS:
            sub = rec[rec["model"] == m]
            w   = sub["task"].value_counts()
            p0  = (null.loc[w.index, outcome] * w).sum() / w.sum()
            p   = binomtest(int(sub[outcome].sum()), len(sub), p0).pvalue
            res.append({"scope": scope, "outcome": OUTCOMES[outcome], "model": m,
                        "rate": sub[outcome].mean(), "null": p0, "p": p})
        for r, pa in zip(res, holm([r["p"] for r in res])):
            r["p_holm"] = pa
        rows += res
    return rows


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    raw   = pd.read_csv(RAW_DATA_PATH)
    raw   = raw[raw["tools"].notna()]
    bench = pd.read_csv(BENCHMARK_FINAL_PATH)
    bench["tool"] = bench["tool"].str.lower().str.strip()

    scored, shares, tests = {}, [], []
    for scope in SCOPES:
        rec, null     = score_scope(raw, bench, scope)
        scored[scope] = rec
        tests        += binomial_tests(rec, null, scope)
        sh = rec.groupby("model")[list(OUTCOMES)].mean().rename(columns=OUTCOMES)
        shares.append(sh.assign(scope=scope).reset_index())

        mt = (rec.groupby(["model", "task", "prompt_type"])[["pipeline_accuracy", "pop_index"]]
                 .mean().unstack())
        n, e = mt["pop_index"]["naive"], mt["pop_index"]["expert"]
        print(f"[{scope:>6}] prompt effect on popularity: {n.mean():.3f} → {e.mean():.3f} "
              f"(Wilcoxon P={wilcoxon(n, e).pvalue:.3g}, n={len(n)})")

    shares = pd.concat(shares)
    tests  = pd.DataFrame(tests)

    key  = ["model", "task", "prompt_type", "run"]
    cols = list(OUTCOMES)
    both = scored["global"].merge(scored["step"], on=key, suffixes=("_g", "_s"))
    kept = (both[[f"{c}_g" for c in cols]].values == both[[f"{c}_s" for c in cols]].values).all(axis=1)
    print(f"\nRecommendations keeping their category, global → step: {kept.mean():.1%}")
    print("Per task:", both.assign(kept=kept).groupby("task")["kept"].mean().round(3).to_dict())

    print("\nLo Acc + Hi Pop vs random choice (Holm-adjusted):")
    print(tests[tests["outcome"] == "Lo Acc + Hi Pop"]
          .pivot(index="model", columns="scope", values="rate").round(3)[SCOPES].to_string())

    os.makedirs(OUT_DIR, exist_ok=True)
    shares.to_csv(os.path.join(OUT_DIR, "popularity_scope_shares.csv"), index=False)
    tests.to_csv(os.path.join(OUT_DIR, "popularity_scope_tests.csv"), index=False)
    print(f"\nSaved → {OUT_DIR}")
