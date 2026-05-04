# data_preprocess.py
#
# The dataset must be a parquet file where each row is one training contract. The key fields
# are `question` (what agents see as the problem), `contract` (the structured data that
# `BiddingEnv.reset()` uses), and `data_source` (used by DrMAS for metrics grouping).
#
#

"""
Generate a synthetic contracts dataset for the bidding scenario.
Run once before training:
  python examples/bidding_trainer/data_preprocess.py
"""

import os
import random
import json
import pandas as pd

random.seed(42)

CONTRACT_TEMPLATES = [
    {
        "sector": "software",
        "description": "Develop a {scale} {type} platform with {features}.",
        "requirements": "React/TypeScript frontend, Python FastAPI backend, PostgreSQL, Docker, CI/CD.",
        "duration": "{months} months",
        "budget_low": 40_000,
        "budget_high": 200_000,
    },
    {
        "sector": "construction",
        "description": "Refurbish {floors} floors of a commercial office building in {city}.",
        "requirements": "Structural survey, HVAC upgrade, electrical rewiring, ADA compliance.",
        "duration": "{months} months",
        "budget_low": 150_000,
        "budget_high": 800_000,
    },
    {
        "sector": "marketing",
        "description": "Design and execute a {duration_label} digital marketing campaign for a {niche} brand.",
        "requirements": "SEO, paid social, content creation, monthly reporting, 3× ROAS target.",
        "duration": "{months} months",
        "budget_low": 20_000,
        "budget_high": 120_000,
    },
    {
        "sector": "data_engineering",
        "description": "Build a real-time {scale} data pipeline ingesting {sources} sources.",
        "requirements": "Apache Kafka, Spark, Snowflake, dbt, monitoring dashboard, SLA 99.9%.",
        "duration": "{months} months",
        "budget_low": 60_000,
        "budget_high": 300_000,
    },
]

SCALES   = ["small", "mid-size", "enterprise-grade"]
TYPES    = ["SaaS", "internal tooling", "e-commerce", "analytics"]
FEATURES = ["real-time notifications, role-based access, and mobile support",
            "AI-powered recommendations and third-party integrations",
            "multi-tenant architecture and advanced reporting"]
CITIES   = ["New York", "Chicago", "San Francisco", "Boston", "Seattle"]
NICHES   = ["health & wellness", "fintech", "direct-to-consumer fashion", "edtech"]


def generate_contract(idx: int) -> dict:
    tmpl   = random.choice(CONTRACT_TEMPLATES)
    months = random.randint(3, 18)
    low    = tmpl["budget_low"]
    high   = tmpl["budget_high"]

    description = tmpl["description"].format(
        scale=random.choice(SCALES),
        type=random.choice(TYPES),
        features=random.choice(FEATURES),
        floors=random.randint(2, 20),
        city=random.choice(CITIES),
        niche=random.choice(NICHES),
        duration_label=f"{months}-month",
        sources=random.randint(3, 15),
        months=months,
    )

    contract = {
        "id":           f"CONTRACT-{idx:05d}",
        "sector":       tmpl["sector"],
        "description":  description,
        "requirements": tmpl["requirements"],
        "budget_range": f"${low:,} – ${high:,}",
        "duration":     tmpl["duration"].format(months=months),
    }

    # `question` is the text shown to agents as the initial env observation
    question = (
        f"Contract ID: {contract['id']}\n"
        f"Description: {contract['description']}\n"
        f"Budget Range: {contract['budget_range']}\n"
        f"Requirements: {contract['requirements']}\n"
        f"Duration: {contract['duration']}"
    )

    return {
        "question":    question,
        "contract":    json.dumps(contract),   # stored as JSON string in parquet
        "data_source": "bidding",
        "ground_truth": "",                    # no ground truth; rewards come from LLM judge
    }


def main():
    os.makedirs(os.path.expanduser("~/data/bidding"), exist_ok=True)

    train_rows = [generate_contract(i)       for i in range(2000)]
    val_rows   = [generate_contract(i + 2000) for i in range(200)]

    train_df = pd.DataFrame(train_rows)
    val_df   = pd.DataFrame(val_rows)

    train_path = os.path.expanduser("~/data/bidding/train.parquet")
    val_path   = os.path.expanduser("~/data/bidding/val.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path,   index=False)

    print(f"Saved {len(train_df)} training contracts  → {train_path}")
    print(f"Saved {len(val_df)}  validation contracts → {val_path}")
    print("\nSample row:")
    print(train_df.iloc[0].to_dict())


if __name__ == "__main__":
    main()

# **Note:** The `contract` column is stored as a JSON string. You must deserialise it in
# your data collation. In `BiddingEnv.reset()` the caller passes `extras["contract"]` as
# a dict, so wherever DrMAS reads the parquet row and calls `env.reset()`, parse the JSON:

# ```python
# import json
# extras["contract"] = json.loads(row["contract"])
# ```

# Check `agent_system/multi_turn_rollout/rollout_loop.py` in `_prepare_gen_batch()` (or
# equivalent) to find where `env.reset()` is called, and add this line there.