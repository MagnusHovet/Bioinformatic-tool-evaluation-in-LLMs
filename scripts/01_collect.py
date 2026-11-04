"""
01_collect.py
=============
Queries each LLM to propose bioinformatic tool pipelines across all
task and prompt-type combinations, and saves raw results to CSV.
 
Each model is queried N_RUNS times per task × prompt combination,
producing a total of N_MODELS × N_TASKS × N_ROLES × N_RUNS rows.
Structured outputs are enforced via Instructor and Pydantic models
defined in config/settings.py.
 
Output
------
  runs/<RUN>/raw/ai_bioinfo_pipeline.csv  (rewritten after each model)

Usage
-----
  python 01_collect.py                                  (rerun, the default)
  $env:RUN = "2026-04_original"; python 01_collect.py   (original run)
  python 01_collect.py RNA-seq                          (recollect one or more tasks only)

Passing task names replaces those tasks' rows for every model and keeps
all other rows; the previous raw file is backed up first.
"""
 
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import time
import instructor
import pandas as pd
from datetime import datetime
 
from config.settings import (
    INITIAL, TASKS, ROLES, MODELS, MODEL_MODES, N_RUNS, RAW_DATA_PATH, TASK_TO_MODEL,
)
 
 
# ── Helpers ───────────────────────────────────────────────────────────

def save(results):
    """Write all results so far to RAW_DATA_PATH."""
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    pd.DataFrame(results).to_csv(RAW_DATA_PATH, index=False)


def served_model(completion):
    """Exact model version reported by the provider, if available."""
    return (getattr(completion, "model", None)
            or getattr(completion, "model_version", None))


def load_completed():
    """
    Resume support: keep rows for models that already have at least 90%
    of their queries answered successfully (occasional validation
    failures are kept as error rows); any other model — e.g. one that
    failed on billing — is collected again from scratch.
    """
    if not os.path.exists(RAW_DATA_PATH):
        return [], set()
    df       = pd.read_csv(RAW_DATA_PATH)
    expected = len(TASKS) * len(ROLES) * N_RUNS
    ok       = df[df["tools"].notna()] if "tools" in df else df.iloc[0:0]
    done     = {m for m, n in ok.groupby("model").size().items() if n >= 0.9 * expected}
    return df[df["model"].isin(done)].to_dict("records"), done


def load_other_tasks(tasks):
    """
    Task-only recollection: back up the raw file, then keep every row
    except those for `tasks`, which are queried again for all models.
    """
    df     = pd.read_csv(RAW_DATA_PATH)
    backup = RAW_DATA_PATH.replace(".csv", f".before_{'_'.join(tasks)}.csv")
    df.to_csv(backup, index=False)
    print(f"Backed up {RAW_DATA_PATH} → {backup}")
    return df[~df["task"].isin(tasks)].to_dict("records"), set()


# ── Collection loop ───────────────────────────────────────────────────

task_filter = sys.argv[1:]
unknown     = set(task_filter) - set(TASKS)
if unknown:
    sys.exit(f"Unknown task(s): {sorted(unknown)}; choose from {list(TASKS)}")

if task_filter:
    results, done = load_other_tasks(task_filter)
    tasks         = {t: TASKS[t] for t in task_filter}
else:
    results, done = load_completed()
    tasks         = TASKS
start_time = time.time()

for model in MODELS:

    if model in done:
        print(f"Skipping {model} — already complete in {RAW_DATA_PATH}")
        continue

    kwargs = {}
    if model in MODEL_MODES:
        kwargs["mode"] = instructor.Mode[MODEL_MODES[model]]
    if model.startswith("deepseek"):
        kwargs["base_url"] = "https://api.deepseek.com"
    client = instructor.from_provider(model, **kwargs)
 
    for task_name, task_prompt in tasks.items():
 
        for prompt_type, prompt_style in ROLES.items():
 
            for run in range(N_RUNS):
 
                elapsed_n  = time.time() - start_time
                print(
                    f"Starting run {run + 1}/{N_RUNS} | {model} | "
                    f"{task_name} | {prompt_type} | "
                    f"elapsed: {elapsed_n:.1f}s ({elapsed_n/60:.1f} min)"
                )
                call_start = time.time()
                prompt     = f"{prompt_style}\n\nTask:\n{task_prompt}"
 
                try:
                    pipeline, completion = client.create_with_completion(
                        response_model=TASK_TO_MODEL[task_name],
                        messages=[
                            {"role": "system", "content": INITIAL},
                            {"role": "user",   "content": prompt},
                        ],
                    )
 
                    call_elapsed = time.time() - call_start
                    print(f"  -> done in {call_elapsed:.1f}s")
 
                    tool_list = pipeline.to_list()  # type: ignore
 
                    results.append({
                        "timestamp":   datetime.now(),
                        "model":       model,
                        "task":        task_name,
                        "prompt_type": prompt_type,
                        "run":         run + 1,
                        "tools":       "|".join(tool_list),   # type: ignore
                        "tool_count":  len(tool_list),        # type: ignore
                        "model_version": served_model(completion),
                    })
 
                except Exception as e:
                    results.append({
                        "timestamp":   datetime.now(),
                        "model":       model,
                        "task":        task_name,
                        "prompt_type": prompt_type,
                        "run":         run + 1,
                        "error":       str(e),
                    })

    save(results)
    print(f"  Saved {len(results)} rows so far → {RAW_DATA_PATH}")


# ── Save ──────────────────────────────────────────────────────────────

elapsed = time.time() - start_time
print(f"Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

save(results)
print(f"Saved {len(results)} rows to {RAW_DATA_PATH}")