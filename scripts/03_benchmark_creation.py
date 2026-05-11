import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

import time
import requests
import pandas as pd

from config.settings import BENCHMARK_MANUAL_PATH, BENCHMARK_FINAL_PATH


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Set GITHUB_TOKEN environment variable")

HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

df = pd.read_csv(BENCHMARK_MANUAL_PATH)

def get_repo_stats(repo):
    url = f"https://api.github.com/repos/{repo}"
    r = requests.get(url, headers=HEADERS)

    if r.status_code == 200:
        data = r.json()
        return {
            "stars": data.get("stargazers_count", 0),
            "forks": data.get("forks_count", 0)
        }

    elif r.status_code == 404:
        print(f"Repo not found: {repo}")
        return {"stars": None, "forks": None}

    elif r.status_code == 403:
        print("Rate limited. Sleeping...")
        time.sleep(60)
        return get_repo_stats(repo)

    else:
        print(f"Error {r.status_code} for {repo}")
        return {"stars": None, "forks": None}


stars = []
forks = []

for repo in df["repo"]:
    if pd.isna(repo):
        stars.append(None)
        forks.append(None)
        continue

    stats = get_repo_stats(repo)
    stars.append(stats["stars"])
    forks.append(stats["forks"])

    time.sleep(0.2)  # avoid hitting rate limit


df["github_stars"] = stars
df["github_forks"] = forks
print(f" Tools without a repo {df["github_stars"].isna().sum()}")


# =========================
# 💾 SAVE
# =========================

os.makedirs(os.path.dirname(BENCHMARK_FINAL_PATH), exist_ok=True)
df.to_csv(BENCHMARK_FINAL_PATH, index=False)

print(f"\nSaved → {BENCHMARK_FINAL_PATH}")