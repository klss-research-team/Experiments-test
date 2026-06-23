"""
Generate procurement auction dataset for DrMAS bidding environment.

Outputs:
  - drmas_bidding/train.parquet  : DrMAS training format
  - drmas_bidding/test.parquet   : DrMAS evaluation format

Schema (new per-round design):
  category              : str   — contract category (fixed per episode)
  task_description      : str   — short episode header shown to agents
  cost_floor            : float — lower bound for per-round cost sampling
  cost_ceiling          : float — upper bound for per-round cost sampling
  budget_multiplier_lo  : float — budget = cost * U(lo, hi) per round
  budget_multiplier_hi  : float
  data_source           : str   — e.g. "bidding_software"

NOTE: The canonical version of this script is:
  DrMAS_bidding_setup1/examples/data_preprocess/drmas_bidding.py
This root-level copy mirrors that schema for quick local testing.
"""

import json
import os
import random
import pandas as pd

SCENARIO_TEMPLATES = [
    ("software",          "Software Development Contract",     (80,  400),  (1.3, 2.2)),
    ("healthcare",        "Healthcare Services Contract",      (150, 1200), (1.2, 1.9)),
    ("construction",      "Construction Contract",             (200, 2000), (1.2, 1.8)),
    ("logistics",         "Logistics Services Contract",       (50,  300),  (1.3, 2.5)),
    ("manufacturing",     "Manufacturing Supply Contract",     (100, 800),  (1.2, 2.0)),
    ("consulting",        "Professional Services Contract",    (60,  500),  (1.4, 2.5)),
    ("energy",            "Energy Services Contract",          (300, 3000), (1.15, 1.7)),
    ("IT infrastructure", "IT Infrastructure Contract",       (100, 900),  (1.3, 2.1)),
    ("legal",             "Legal Services Contract",           (50,  600),  (1.4, 2.6)),
    ("marketing",         "Marketing Services Contract",       (40,  350),  (1.5, 3.0)),
]


def sample_scenario(rng: random.Random) -> dict:
    category, task_description, cost_range, bm_range = rng.choice(SCENARIO_TEMPLATES)
    return {
        "category":             category,
        "task_description":     task_description,
        "cost_floor":           float(cost_range[0]),
        "cost_ceiling":         float(cost_range[1]),
        "budget_multiplier_lo": bm_range[0],
        "budget_multiplier_hi": bm_range[1],
    }


def build_drmas_row(scenario: dict, idx: int, split: str) -> dict:
    td  = scenario["task_description"]
    cat = scenario["category"]
    return {
        "data_source":  "bidding",
        "ability":      "bidding",
        "prompt":       json.dumps([{"role": "user", "content": td}]),
        "reward_model": json.dumps({"style": "rule", "ground_truth": None}),
        "extra_info":   json.dumps({"split": split, "index": idx, "category": cat}),
        "env_kwargs":   json.dumps({
            "category":             cat,
            "task_description":     td,
            "cost_floor":           scenario["cost_floor"],
            "cost_ceiling":         scenario["cost_ceiling"],
            "budget_multiplier_lo": scenario["budget_multiplier_lo"],
            "budget_multiplier_hi": scenario["budget_multiplier_hi"],
            "question":             td,
            "data_source":          f"bidding_{cat.replace(' ', '_')}",
        }),
    }


def generate(n: int, split: str, seed: int) -> list:
    rng = random.Random(seed)
    return [sample_scenario(rng) for _ in range(n)]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--test_size",  type=int, default=500)
    parser.add_argument("--out_dir",    type=str,
                        default=os.path.join(os.path.dirname(__file__), "drmas_bidding"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Generating {args.train_size} train + {args.test_size} test scenarios...")
    train_scenarios = generate(args.train_size, "train", seed=42)
    test_scenarios  = generate(args.test_size,  "test",  seed=1337)

    train_rows = [build_drmas_row(s, i, "train") for i, s in enumerate(train_scenarios)]
    test_rows  = [build_drmas_row(s, i, "test")  for i, s in enumerate(test_scenarios)]

    train_path = os.path.join(args.out_dir, "train.parquet")
    test_path  = os.path.join(args.out_dir, "test.parquet")
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(test_rows).to_parquet(test_path,   index=False)
    print(f"Train parquet → {train_path}  ({args.train_size} rows)")
    print(f"Test parquet  → {test_path}   ({args.test_size} rows)")

    row = train_scenarios[0]
    print(f"\nSample scenario[0]:")
    print(f"  category         : {row['category']}")
    print(f"  task_description : {row['task_description']}")
    print(f"  cost_floor       : ${row['cost_floor']:.2f}")
    print(f"  cost_ceiling     : ${row['cost_ceiling']:.2f}")
    print(f"  budget_mult      : {row['budget_multiplier_lo']}–{row['budget_multiplier_hi']}")
