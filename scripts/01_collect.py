import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

import time
import instructor
import pandas as pd
from pydantic import BaseModel, field_validator
from typing import List, Optional
from datetime import datetime

from config.settings import INITIAL, TASKS, ROLES, MODELS, N_RUNS, RAW_DATA_PATH, TASK_TO_MODEL


#Schema for data handling to LLM

results = []
start_time = time.time()

for model in MODELS: #loops through each model
    if model.startswith("deepseek"):
        client = instructor.from_provider(model, base_url="https://api.deepseek.com")
    else:
        client = instructor.from_provider(model)

    for task_name, task_prompt in TASKS.items(): #loops through each Task question

        for prompt_type, prompt_style in ROLES.items(): #loops through each Prompt style

            for run in range(N_RUNS):
                
                elapsed_n = time.time() - start_time
                print(f"Starting run {run + 1}/{N_RUNS} | {model} | {task_name} | {prompt_type} | elapsed: {elapsed_n:.1f}s ({elapsed_n/60:.1f} min)")
                call_start = time.time()

                prompt = f"{prompt_style}\n\nTask:\n{task_prompt}"
                try:
                    pipeline = client.create(
                    response_model=TASK_TO_MODEL[task_name],
                        messages=[
                            {"role": "system", "content": INITIAL},
                            {"role": "user", "content": prompt}
                            ],
                    )
                    
                    call_elapsed = time.time() - call_start
                    print(f"  -> done in {call_elapsed:.1f}s")
                    
                    tool_list = pipeline.to_list() # type: ignore 
                    
                    results.append({
                        "timestamp": datetime.now(), 
                        "model": model,
                        "task": task_name,
                        "prompt_type": prompt_type,
                        "run": run + 1, 
                        "tools": "|".join(tool_list), # type: ignore 
                        "tool_count": len(tool_list), # type: ignore                     
                    })
                except Exception as e:

                    results.append({
                        "timestamp": datetime.now(),
                        "model": model,
                        "task": task_name,
                        "prompt_type": prompt_type,
                        "run": run + 1,
                        "error": str(e)
                    })


elapsed = time.time() - start_time
print(f"Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

df = pd.DataFrame(results)
os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
df.to_csv(RAW_DATA_PATH, index=False)
print(f"Saved {len(df)} rows to {RAW_DATA_PATH}")
