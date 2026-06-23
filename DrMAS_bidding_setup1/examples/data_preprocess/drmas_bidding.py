"""
Synthetic dataset generator for the DrMAS procurement bidding environment.

Each dataset row fixes the contract CATEGORY for an episode (e.g. "software").
Within the episode, BiddingEnv samples a fresh scenario description, cost, and
budget at the start of every round, giving agents meaningful per-round context
to reason about rather than just reacting to numbers.

Usage:
    python drmas_bidding.py --local_dir ~/data/drmas_bidding
    python drmas_bidding.py --local_dir ~/data/drmas_bidding --train_size 5000 --test_size 500
"""

import argparse
import json
import os
import random

import datasets
from datasets import Dataset


# ---------------------------------------------------------------------------
# Contract category templates
# Each entry: (category, episode_header, cost_range, budget_multiplier_range)
# episode_header        : shown to agents every round as the contract type context.
# cost_range            : (cost_floor, cost_ceiling) passed to BiddingEnv for
#                         per-round cost sampling.
# budget_multiplier_range: (lo, hi) — budget = cost * U(lo, hi) each round.
# ---------------------------------------------------------------------------
SCENARIO_TEMPLATES = [
    (
        "software",
        "Software Development Contract",
        (80, 400),
        (1.3, 2.2),
    ),
    (
        "construction",
        "Construction Contract",
        (200, 2000),
        (1.2, 1.8),
    ),
    (
        "logistics",
        "Logistics Services Contract",
        (50, 300),
        (1.3, 2.5),
    ),
    (
        "manufacturing",
        "Manufacturing Supply Contract",
        (100, 800),
        (1.2, 2.0),
    ),
    (
        "consulting",
        "Professional Services Contract",
        (60, 500),
        (1.4, 2.5),
    ),
]


def _sample_scenario(rng: random.Random) -> dict:
    """Pick a contract category for an episode.

    Per-round scenario descriptions, costs, and budgets are sampled by
    BiddingEnv at runtime using the category-level bounds returned here.
    """
    category, episode_header, cost_range, bm_range = rng.choice(SCENARIO_TEMPLATES)

    return {
        "category": category,
        "task_description": episode_header,
        "cost_floor": float(cost_range[0]),
        "cost_ceiling": float(cost_range[1]),
        "budget_multiplier_lo": float(bm_range[0]),
        "budget_multiplier_hi": float(bm_range[1]),
    }


def build_row(scenario: dict, idx: int, split: str) -> dict:
    """
    Format a scenario dict into the row structure expected by the training loop.

    The training loop pops `env_kwargs` from the batch and passes it directly
    to BiddingMultiProcessEnv.reset(), so all fields BiddingEnv.reset() reads
    must be present there.
    """
    category = scenario["category"]
    task_description = scenario["task_description"]
    cost_floor = scenario["cost_floor"]
    cost_ceiling = scenario["cost_ceiling"]
    budget_multiplier_lo = scenario["budget_multiplier_lo"]
    budget_multiplier_hi = scenario["budget_multiplier_hi"]

    return {
        "data_source": "bidding",
        "ability": "bidding",
        "prompt": [
            {
                "role": "user",
                "content": task_description,
            }
        ],
        "reward_model": {
            "style": "rule",
            "ground_truth": None,
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "category": category,
            "cost_floor": cost_floor,
            "cost_ceiling": cost_ceiling,
            "budget_multiplier_lo": budget_multiplier_lo,
            "budget_multiplier_hi": budget_multiplier_hi,
        },
        # env_kwargs is passed verbatim to BiddingMultiProcessEnv.reset()
        "env_kwargs": {
            "category": category,
            "task_description": task_description,
            "cost_floor": cost_floor,
            "cost_ceiling": cost_ceiling,
            "budget_multiplier_lo": budget_multiplier_lo,
            "budget_multiplier_hi": budget_multiplier_hi,
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
            "category":             str(raw_row["category"]),
            "task_description":     str(raw_row.get("task_description", raw_row["category"].replace("_", " ").title())),
            "cost_floor":           float(raw_row["cost_floor"]),
            "cost_ceiling":         float(raw_row["cost_ceiling"]),
            "budget_multiplier_lo": float(raw_row["budget_multiplier_lo"]),
            "budget_multiplier_hi": float(raw_row["budget_multiplier_hi"]),
        },
        idx,
        split,
    )


def load_from_local(path: str, test_ratio: float = 0.1, seed: int = 42):
    """
    Load a dataset from a local JSON or JSONL file.
    Expected fields per row: category, cost_floor, cost_ceiling,
    budget_multiplier_lo, budget_multiplier_hi. task_description is optional.
    Returns (train_dataset, test_dataset).
    """
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            raw = json.load(f)
        else:
            raw = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(raw)} rows from {path}")

    required = ("category", "cost_floor", "cost_ceiling", "budget_multiplier_lo", "budget_multiplier_hi")
    valid, skipped = [], 0
    for i, row in enumerate(raw):
        missing = [c for c in required if c not in row or row[c] is None]
        if missing:
            print(f"  [skip] row {i}: missing fields {missing}")
            skipped += 1
            continue
        if float(row["cost_ceiling"]) <= float(row["cost_floor"]):
            print(f"  [skip] row {i}: cost_ceiling must be > cost_floor")
            skipped += 1
            continue
        if float(row["budget_multiplier_hi"]) <= float(row["budget_multiplier_lo"]):
            print(f"  [skip] row {i}: budget_multiplier_hi must be > budget_multiplier_lo")
            skipped += 1
            continue
        valid.append(dict(row))

    print(f"Validation complete: {len(valid)} valid rows, {skipped} skipped.")
    if not valid:
        raise ValueError(
            "No valid rows found. Each row must have: "
            "category, cost_floor, cost_ceiling, budget_multiplier_lo, budget_multiplier_hi."
        )

    rng = random.Random(seed)
    rng.shuffle(valid)
    split_at = max(1, int(len(valid) * (1 - test_ratio)))
    train_rows = valid[:split_at]
    test_rows  = valid[split_at:]

    train_ds = Dataset.from_list([_hub_to_row(r, i, "train") for i, r in enumerate(train_rows)])
    test_ds  = Dataset.from_list([_hub_to_row(r, i, "test")  for i, r in enumerate(test_rows)])
    return train_ds, test_ds


def load_from_hub(dataset_id: str, test_ratio: float = 0.1, seed: int = 42):
    """
    Load a dataset from HuggingFace Hub, validate, split, and convert.
    Expected columns: category, cost_floor, cost_ceiling,
    budget_multiplier_lo, budget_multiplier_hi.
    Returns (train_dataset, test_dataset).
    """
    raw = datasets.load_dataset(dataset_id, split="train")

    required = ("category", "cost_floor", "cost_ceiling", "budget_multiplier_lo", "budget_multiplier_hi")
    valid, skipped = [], 0
    for i, row in enumerate(raw):
        missing = [c for c in required if c not in row or row[c] is None]
        if missing:
            print(f"  [skip] row {i}: missing columns {missing}")
            skipped += 1
            continue
        if float(row["cost_ceiling"]) <= float(row["cost_floor"]):
            print(f"  [skip] row {i}: cost_ceiling must be > cost_floor")
            skipped += 1
            continue
        valid.append(dict(row))

    print(f"Validation complete: {len(valid)} valid rows, {skipped} skipped.")
    if not valid:
        raise ValueError(
            "No valid rows found. Check that your dataset has columns: "
            "category, cost_floor, cost_ceiling, budget_multiplier_lo, budget_multiplier_hi."
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
        "--local_file", default=None,
        help="Path to a local JSON or JSONL file. "
             "Required fields: category, cost_floor, cost_ceiling, "
             "budget_multiplier_lo, budget_multiplier_hi.",
    )
    parser.add_argument(
        "--dataset_id", default=None,
        help="HuggingFace dataset ID.",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1,
        help="Fraction of rows to use as the test split (default: 0.1).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the train/test split (default: 42).",
    )
    parser.add_argument("--local_dir",   default="~/data/drmas_bidding")
    parser.add_argument("--train_size",  type=int, default=5000, help="Synthetic generator only.")
    parser.add_argument("--test_size",   type=int, default=500,  help="Synthetic generator only.")
    parser.add_argument("--train_seed",  type=int, default=42,   help="Synthetic generator only.")
    parser.add_argument("--test_seed",   type=int, default=1337, help="Synthetic generator only.")
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    if args.local_file:
        print(f"Loading dataset from local file: {args.local_file}")
        train_dataset, test_dataset = load_from_local(args.local_file, args.test_ratio, args.seed)
    elif args.dataset_id:
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
    print(f"  category          : {row['extra_info']['category']}")
    print(f"  cost_floor        : ${row['env_kwargs']['cost_floor']:.2f}")
    print(f"  cost_ceiling      : ${row['env_kwargs']['cost_ceiling']:.2f}")
    print(f"  budget_mult range : {row['env_kwargs']['budget_multiplier_lo']:.1f}x – {row['env_kwargs']['budget_multiplier_hi']:.1f}x")
    print(f"  episode header    : {row['prompt'][0]['content']}")
