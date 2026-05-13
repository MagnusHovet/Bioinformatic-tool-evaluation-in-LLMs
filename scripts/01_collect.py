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
  data/raw/ai_bioinfo_pipeline.csv
 
Usage
-----
  python 01_collect.py
"""
 
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import time
import instructor
import pandas as pd
from datetime import datetime
 
from config.settings import INITIAL, TASKS, ROLES, MODELS, N_RUNS, RAW_DATA_PATH, TASK_TO_MODEL
 
 
# ── Collection loop ───────────────────────────────────────────────────
 
results    = []
start_time = time.time()
 
for model in MODELS:
 
    if model.startswith("deepseek"):
        client = instructor.from_provider(model, base_url="https://api.deepseek.com")
    else:
        client = instructor.from_provider(model)
 
    for task_name, task_prompt in TASKS.items():
 
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
                    pipeline = client.create(
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
 
 
# ── Save ──────────────────────────────────────────────────────────────
 
elapsed = time.time() - start_time
print(f"Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
 
df = pd.DataFrame(results)
os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
df.to_csv(RAW_DATA_PATH, index=False)
print(f"Saved {len(df)} rows to {RAW_DATA_PATH}")