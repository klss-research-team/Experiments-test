"""
Synthetic dataset generator for the DrMAS procurement bidding environment.

Generates parameterized auction scenarios across contract categories with
varied cost and budget values. No external dataset download required.

Usage:
    python drmas_bidding.py --local_dir ~/data/drmas_bidding
    python drmas_bidding.py --local_dir ~/data/drmas_bidding --train_size 5000 --test_size 500
"""

import argparse
import os
import random

import datasets
from datasets import Dataset


# ---------------------------------------------------------------------------
# Contract scenario templates
# Each entry: (category, description_template, cost_range, budget_multiplier_range)
# cost_range       : (min_cost, max_cost) in dollars
# budget_multiplier: budget = cost * U(lo, hi)  — buyer always has headroom
# ---------------------------------------------------------------------------
SCENARIO_TEMPLATES = [
    (
        "software",
        (
            "Software development contract: build a {adjective} {product} system. "
            "Estimated development cost: ${cost:.0f}. Buyer budget ceiling: ${budget:.0f}."
        ),
        (80, 400),
        (1.3, 2.2),
        ["web", "mobile", "data pipeline", "API integration", "dashboard", "automation"],
        ["enterprise", "scalable", "real-time", "cloud-native", "AI-powered", "modular"],
    ),
    (
        "construction",
        (
            "Construction contract: {adjective} {product} project. "
            "Estimated build cost: ${cost:.0f}. Buyer budget ceiling: ${budget:.0f}."
        ),
        (200, 2000),
        (1.2, 1.8),
        ["office renovation", "warehouse expansion", "facility upgrade", "road repair", "bridge maintenance", "drainage installation"],
        ["small-scale", "medium-scale", "high-priority", "time-sensitive", "multi-phase", "turnkey"],
    ),
    (
        "logistics",
        (
            "Logistics services contract: {adjective} {product} operation. "
            "Estimated operational cost: ${cost:.0f}. Buyer budget ceiling: ${budget:.0f}."
        ),
        (50, 300),
        (1.3, 2.5),
        ["freight forwarding", "last-mile delivery", "cold-chain transport", "customs clearance", "warehousing", "distribution"],
        ["regional", "national", "cross-border", "express", "bulk", "scheduled"],
    ),
    (
        "manufacturing",
        (
            "Manufacturing supply contract: {adjective} production of {product}. "
            "Estimated unit production cost: ${cost:.0f}. Buyer budget ceiling: ${budget:.0f}."
        ),
        (100, 800),
        (1.2, 2.0),
        ["electronic components", "mechanical parts", "packaging materials", "industrial equipment", "textile goods", "plastic components"],
        ["high-volume", "custom", "precision", "certified", "rapid", "batch"],
    ),
    (
        "consulting",
        (
            "Professional services contract: {adjective} {product} engagement. "
            "Estimated consulting cost: ${cost:.0f}. Buyer budget ceiling: ${budget:.0f}."
        ),
        (60, 500),
        (1.4, 2.5),
        ["strategy consulting", "IT advisory", "legal review", "financial audit", "compliance assessment", "market research"],
        ["short-term", "ongoing", "project-based", "retainer", "diagnostic", "implementation"],
    ),
]


def _sample_scenario(rng: random.Random) -> dict:
    """Draw one random contract scenario and return its parameters."""
    category, template, cost_range, bm_range, products, adjectives = rng.choice(
        SCENARIO_TEMPLATES
    )

    cost = round(rng.uniform(*cost_range), 2)
    budget_multiplier = rng.uniform(*bm_range)
    budget = round(cost * budget_multiplier, 2)

    product = rng.choice(products)
    adjective = rng.choice(adjectives)

    task_description = template.format(
        adjective=adjective,
        product=product,
        cost=cost,
        budget=budget,
    )

    return {
        "category": category,
        "cost": cost,
        "budget": budget,
        "task_description": task_description,
    }


def build_row(scenario: dict, idx: int, split: str) -> dict:
    """
    Format a scenario dict into the row structure expected by the training loop.

    The training loop pops `env_kwargs` from the batch and passes it directly
    to BiddingMultiProcessEnv.reset(), so all fields BiddingEnv.reset() reads
    must be present there.
    """
    task_description = scenario["task_description"]
    cost = scenario["cost"]
    budget = scenario["budget"]
    category = scenario["category"]

    return {
        "data_source": "bidding",
        "ability": "bidding",
        # prompt is tokenized as the initial user turn for the LLM
        "prompt": [
            {
                "role": "user",
                "content": task_description,
            }
        ],
        # reward_model is required by the DataProto schema; unused for bidding
        # (rewards are computed inside BiddingEnv.step, not from a reward model)
        "reward_model": {
            "style": "rule",
            "ground_truth": None,
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "cost": cost,
            "budget": budget,
            "category": category,
        },
        # env_kwargs is passed verbatim to BiddingMultiProcessEnv.reset()
        "env_kwargs": {
            "cost": cost,
            "budget": budget,
            "question": task_description,
            "task_description": task_description,
            "data_source": f"bidding_{category}",
        },
    }


def generate_split(n: int, split: str, seed: int) -> Dataset:
    rng = random.Random(seed)
    rows = [
        build_row(_sample_scenario(rng), idx, split)
        for idx in range(n)
    ]
    return Dataset.from_list(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DrMAS bidding dataset")
    parser.add_argument("--local_dir", default="~/data/drmas_bidding")
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--train_seed", type=int, default=42)
    parser.add_argument("--test_seed", type=int, default=1337)
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Generating {args.train_size} train scenarios (seed={args.train_seed})...")
    train_dataset = generate_split(args.train_size, "train", args.train_seed)

    print(f"Generating {args.test_size} test scenarios (seed={args.test_seed})...")
    test_dataset = generate_split(args.test_size, "test", args.test_seed)

    train_path = os.path.join(local_dir, "train.parquet")
    test_path = os.path.join(local_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"Train: {len(train_dataset)} rows → {train_path}")
    print(f"Test:  {len(test_dataset)} rows  → {test_path}")

    # Quick sanity check on first row
    row = train_dataset[0]
    print("\nSample row[0]:")
    print(f"  category   : {row['extra_info']['category']}")
    print(f"  cost       : ${row['env_kwargs']['cost']:.2f}")
    print(f"  budget     : ${row['env_kwargs']['budget']:.2f}")
    print(f"  description: {row['prompt'][0]['content']}")
