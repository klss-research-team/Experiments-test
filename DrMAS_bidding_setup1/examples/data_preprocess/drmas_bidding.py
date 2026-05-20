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
            "Contract cost range across rounds: ${min_cost:.0f}–${max_cost:.0f}. "
            "Buyer budget ceiling: ${budget:.0f}. "
            "You will be told the exact cost for each round before bidding."
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
            "Contract cost range across rounds: ${min_cost:.0f}–${max_cost:.0f}. "
            "Buyer budget ceiling: ${budget:.0f}. "
            "You will be told the exact cost for each round before bidding."
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
            "Contract cost range across rounds: ${min_cost:.0f}–${max_cost:.0f}. "
            "Buyer budget ceiling: ${budget:.0f}. "
            "You will be told the exact cost for each round before bidding."
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
            "Contract cost range across rounds: ${min_cost:.0f}–${max_cost:.0f}. "
            "Buyer budget ceiling: ${budget:.0f}. "
            "You will be told the exact cost for each round before bidding."
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
            "Contract cost range across rounds: ${min_cost:.0f}–${max_cost:.0f}. "
            "Buyer budget ceiling: ${budget:.0f}. "
            "You will be told the exact cost for each round before bidding."
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

    min_cost, max_cost = cost_range
    budget_multiplier = rng.uniform(*bm_range)
    # Budget ceiling is based on max_cost so all per-round costs fit under budget.
    budget = round(max_cost * budget_multiplier, 2)

    product = rng.choice(products)
    adjective = rng.choice(adjectives)

    task_description = template.format(
        adjective=adjective,
        product=product,
        min_cost=min_cost,
        max_cost=max_cost,
        budget=budget,
    )

    return {
        "category": category,
        "min_cost": float(min_cost),
        "max_cost": float(max_cost),
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
    min_cost = scenario["min_cost"]
    max_cost = scenario["max_cost"]
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
            "min_cost": min_cost,
            "max_cost": max_cost,
            "budget": budget,
            "category": category,
        },
        # env_kwargs is passed verbatim to BiddingMultiProcessEnv.reset()
        "env_kwargs": {
            "min_cost": min_cost,
            "max_cost": max_cost,
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


def _hub_to_row(raw_row: dict, idx: int, split: str) -> dict:
    """Convert a raw HuggingFace dataset row into the training loop format."""
    return build_row(
        {
            "category":         str(raw_row["category"]),
            "min_cost":         float(raw_row["min_cost"]),
            "max_cost":         float(raw_row["max_cost"]),
            "budget":           float(raw_row["budget"]),
            "task_description": str(raw_row["task_description"]),
        },
        idx,
        split,
    )


def load_from_hub(dataset_id: str, test_ratio: float = 0.1, seed: int = 42):
    """
    Load a dataset from HuggingFace Hub, validate, split, and convert.

    Expected columns: task_description, min_cost, max_cost, budget, category.
    Invalid rows are skipped with a printed warning.
    Returns (train_dataset, test_dataset).
    """
    raw = datasets.load_dataset(dataset_id, split="train")

    required = ("task_description", "min_cost", "max_cost", "budget", "category")
    valid, skipped = [], 0
    for i, row in enumerate(raw):
        missing = [c for c in required if c not in row or row[c] is None]
        if missing:
            print(f"  [skip] row {i}: missing columns {missing}")
            skipped += 1
            continue
        if float(row["budget"]) <= float(row["max_cost"]):
            print(f"  [skip] row {i}: budget ({row['budget']}) must be > max_cost ({row['max_cost']})")
            skipped += 1
            continue
        if float(row["max_cost"]) <= float(row["min_cost"]):
            print(f"  [skip] row {i}: max_cost ({row['max_cost']}) must be > min_cost ({row['min_cost']})")
            skipped += 1
            continue
        valid.append(dict(row))

    print(f"Validation complete: {len(valid)} valid rows, {skipped} skipped.")
    if not valid:
        raise ValueError(
            "No valid rows found. Check that your dataset has columns: "
            "task_description, min_cost, max_cost, budget, category — "
            "and that budget > max_cost > min_cost for every row."
        )

    rng = random.Random(seed)
    rng.shuffle(valid)
    split_at = max(1, int(len(valid) * (1 - test_ratio)))
    train_rows = valid[:split_at]
    test_rows  = valid[split_at:]

    train_ds = Dataset.from_list([_hub_to_row(r, i, "train") for i, r in enumerate(train_rows)])
    test_ds  = Dataset.from_list([_hub_to_row(r, i, "test")  for i, r in enumerate(test_rows)])
    return train_ds, test_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DrMAS bidding dataset for training")
    parser.add_argument(
        "--dataset_id", default=None,
        help="HuggingFace dataset ID, e.g. 'your-org/bidding-dataset'. "
             "When set, loads real data from the Hub instead of generating synthetic scenarios.",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1,
        help="Fraction of rows to use as the test split when loading from Hub (default: 0.1 = 10%%).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the train/test split when loading from Hub (default: 42).",
    )
    parser.add_argument("--local_dir",   default="~/data/drmas_bidding")
    parser.add_argument("--train_size",  type=int, default=5000, help="Synthetic generator only.")
    parser.add_argument("--test_size",   type=int, default=500,  help="Synthetic generator only.")
    parser.add_argument("--train_seed",  type=int, default=42,   help="Synthetic generator only.")
    parser.add_argument("--test_seed",   type=int, default=1337, help="Synthetic generator only.")
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    if args.dataset_id:
        print(f"Loading dataset from HuggingFace Hub: {args.dataset_id}")
        train_dataset, test_dataset = load_from_hub(args.dataset_id, args.test_ratio, args.seed)
    else:
        print(f"Generating {args.train_size} synthetic train scenarios (seed={args.train_seed})...")
        train_dataset = generate_split(args.train_size, "train", args.train_seed)
        print(f"Generating {args.test_size} synthetic test scenarios (seed={args.test_seed})...")
        test_dataset = generate_split(args.test_size, "test", args.test_seed)

    train_path = os.path.join(local_dir, "train.parquet")
    test_path  = os.path.join(local_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"Train: {len(train_dataset)} rows → {train_path}")
    print(f"Test:  {len(test_dataset)} rows  → {test_path}")

    # Quick sanity check on first row
    row = train_dataset[0]
    print("\nSample row[0]:")
    print(f"  category   : {row['extra_info']['category']}")
    print(f"  min_cost   : ${row['env_kwargs']['min_cost']:.2f}")
    print(f"  max_cost   : ${row['env_kwargs']['max_cost']:.2f}")
    print(f"  budget     : ${row['env_kwargs']['budget']:.2f}")
    print(f"  description: {row['prompt'][0]['content']}")
