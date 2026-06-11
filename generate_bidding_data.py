"""
Generate procurement auction dataset for DrMAS bidding environment.

Outputs:
  - auction_scenarios.jsonl          : raw schema (task_description, category, min_cost, max_cost, budget)
  - drmas_bidding/train.parquet      : DrMAS training format (5 000 rows)
  - drmas_bidding/test.parquet       : DrMAS evaluation format (500 rows)

Raw schema (matching the project spec):
  task_description : str   — natural language description of the auction scenario
  category         : str   — e.g. "software", "healthcare"
  min_cost         : float — minimum possible cost per round, in dollars
  max_cost         : float — maximum possible cost per round; must be > min_cost
  budget           : float — buyer's budget ceiling; must be > max_cost
"""

import json
import os
import random
import pandas as pd

# ---------------------------------------------------------------------------
# Scenario templates
# Each entry: (category, description_template, cost_range, budget_multiplier_range, products, adjectives)
# budget = max_cost * U(bm_lo, bm_hi)  — guarantees budget > max_cost > min_cost
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
        ["web portal", "mobile app", "data pipeline", "API integration", "dashboard", "automation suite"],
        ["enterprise", "scalable", "real-time", "cloud-native", "AI-powered", "modular"],
    ),
    (
        "healthcare",
        (
            "Healthcare services contract: {adjective} {product}. "
            "Contract cost range across rounds: ${min_cost:.0f}–${max_cost:.0f}. "
            "Buyer budget ceiling: ${budget:.0f}. "
            "You will be told the exact cost for each round before bidding."
        ),
        (150, 1200),
        (1.2, 1.9),
        ["medical equipment supply", "clinical staffing", "lab testing service", "telehealth platform",
         "patient records system", "pharmacy management"],
        ["certified", "HIPAA-compliant", "multi-site", "outpatient", "integrated", "specialized"],
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
        ["office renovation", "warehouse expansion", "facility upgrade", "road repair",
         "bridge maintenance", "drainage installation"],
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
        ["freight forwarding", "last-mile delivery", "cold-chain transport", "customs clearance",
         "warehousing", "distribution network"],
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
        ["electronic components", "mechanical parts", "packaging materials", "industrial equipment",
         "textile goods", "plastic components"],
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
        ["strategy consulting", "IT advisory", "legal review", "financial audit",
         "compliance assessment", "market research"],
        ["short-term", "ongoing", "project-based", "retainer", "diagnostic", "implementation"],
    ),
    (
        "energy",
        (
            "Energy services contract: {adjective} {product}. "
            "Contract cost range across rounds: ${min_cost:.0f}–${max_cost:.0f}. "
            "Buyer budget ceiling: ${budget:.0f}. "
            "You will be told the exact cost for each round before bidding."
        ),
        (300, 3000),
        (1.15, 1.7),
        ["solar panel installation", "HVAC maintenance", "electrical grid upgrade",
         "natural gas supply", "energy audit", "renewable energy certificate"],
        ["large-scale", "commercial", "industrial", "government", "retrofit", "greenfield"],
    ),
    (
        "IT infrastructure",
        (
            "IT infrastructure contract: {adjective} {product} deployment. "
            "Contract cost range across rounds: ${min_cost:.0f}–${max_cost:.0f}. "
            "Buyer budget ceiling: ${budget:.0f}. "
            "You will be told the exact cost for each round before bidding."
        ),
        (100, 900),
        (1.3, 2.1),
        ["server rack", "network switch upgrade", "cloud migration", "cybersecurity solution",
         "backup system", "data centre colocation"],
        ["on-premises", "hybrid", "managed", "enterprise-grade", "high-availability", "zero-trust"],
    ),
    (
        "legal",
        (
            "Legal services contract: {adjective} {product}. "
            "Contract cost range across rounds: ${min_cost:.0f}–${max_cost:.0f}. "
            "Buyer budget ceiling: ${budget:.0f}. "
            "You will be told the exact cost for each round before bidding."
        ),
        (50, 600),
        (1.4, 2.6),
        ["contract drafting", "IP portfolio management", "litigation support", "regulatory filing",
         "due diligence review", "employment law advisory"],
        ["specialized", "full-service", "interim", "on-call", "cross-jurisdictional", "boutique"],
    ),
    (
        "marketing",
        (
            "Marketing services contract: {adjective} {product} campaign. "
            "Contract cost range across rounds: ${min_cost:.0f}–${max_cost:.0f}. "
            "Buyer budget ceiling: ${budget:.0f}. "
            "You will be told the exact cost for each round before bidding."
        ),
        (40, 350),
        (1.5, 3.0),
        ["digital advertising", "brand strategy", "SEO optimization", "content production",
         "social media management", "influencer outreach"],
        ["performance-based", "omnichannel", "data-driven", "targeted", "creative", "integrated"],
    ),
]


def sample_scenario(rng: random.Random) -> dict:
    """Draw one random contract scenario."""
    category, template, cost_range, bm_range, products, adjectives = rng.choice(SCENARIO_TEMPLATES)
    min_cost, max_cost = float(cost_range[0]), float(cost_range[1])
    budget = round(max_cost * rng.uniform(*bm_range), 2)
    product = rng.choice(products)
    adjective = rng.choice(adjectives)
    task_description = template.format(
        adjective=adjective, product=product,
        min_cost=min_cost, max_cost=max_cost, budget=budget,
    )
    return {
        "task_description": task_description,
        "category": category,
        "min_cost": min_cost,
        "max_cost": max_cost,
        "budget": budget,
    }


def build_drmas_row(scenario: dict, idx: int, split: str) -> dict:
    """Wrap raw scenario into the DrMAS training-loop format."""
    td = scenario["task_description"]
    return {
        "data_source": "bidding",
        "ability": "bidding",
        "prompt": json.dumps([{"role": "user", "content": td}]),
        "reward_model": json.dumps({"style": "rule", "ground_truth": None}),
        "extra_info": json.dumps({
            "split": split, "index": idx,
            "min_cost": scenario["min_cost"], "max_cost": scenario["max_cost"],
            "budget": scenario["budget"], "category": scenario["category"],
        }),
        "env_kwargs": json.dumps({
            "min_cost": scenario["min_cost"],
            "max_cost": scenario["max_cost"],
            "budget": scenario["budget"],
            "question": td,
            "task_description": td,
            "data_source": f"bidding_{scenario['category'].replace(' ', '_')}",
        }),
    }


def generate(n: int, split: str, seed: int) -> list:
    rng = random.Random(seed)
    return [sample_scenario(rng) for _ in range(n)]


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "drmas_bidding")
    os.makedirs(out_dir, exist_ok=True)

    TRAIN_N, TEST_N = 5000, 500

    print(f"Generating {TRAIN_N} train + {TEST_N} test scenarios...")
    train_scenarios = generate(TRAIN_N, "train", seed=42)
    test_scenarios  = generate(TEST_N,  "test",  seed=1337)

    # --- Raw JSONL (schema from the image) ---
    raw_path = os.path.join(os.path.dirname(__file__), "auction_scenarios.jsonl")
    with open(raw_path, "w") as f:
        for s in train_scenarios + test_scenarios:
            f.write(json.dumps(s) + "\n")
    print(f"Raw JSONL  → {raw_path}  ({TRAIN_N + TEST_N} rows)")

    # --- DrMAS parquet (train / test) ---
    train_rows = [build_drmas_row(s, i, "train") for i, s in enumerate(train_scenarios)]
    test_rows  = [build_drmas_row(s, i, "test")  for i, s in enumerate(test_scenarios)]

    train_df = pd.DataFrame(train_rows)
    test_df  = pd.DataFrame(test_rows)

    train_path = os.path.join(out_dir, "train.parquet")
    test_path  = os.path.join(out_dir, "test.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path,  index=False)
    print(f"Train parquet → {train_path}  ({len(train_df)} rows)")
    print(f"Test parquet  → {test_path}   ({len(test_df)} rows)")

    # --- Sanity check ---
    row = train_scenarios[0]
    print(f"\nSample scenario[0]:")
    print(f"  category   : {row['category']}")
    print(f"  min_cost   : ${row['min_cost']:.2f}")
    print(f"  max_cost   : ${row['max_cost']:.2f}")
    print(f"  budget     : ${row['budget']:.2f}")
    print(f"  description: {row['task_description']}")

    cats = {}
    for s in train_scenarios:
        cats[s["category"]] = cats.get(s["category"], 0) + 1
    print("\nCategory distribution (train):")
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat:<20} {cnt:>5} rows  ({cnt/TRAIN_N*100:.1f}%)")
