"""
Train a bidding agent using GRPO on the pre-collected auction_rl_dataset.jsonl.

Dataset structure (3000 rows, 100 unique rounds):
  - round          : int 1-100; rows with the same round share an identical user_prompt
  - system_prompt  : shared system message
  - user_prompt    : context (round #, bid/score history) — same for all rows in a round
  - response       : BID: <int>\nREASONING: <text>
  - reward         : float 1-10 (judge score)
  - strategy_type  : aggressive / strong / moderate / weak / collusive / mismatched_*

GRPO requires groups: multiple responses to the *same* prompt, each with a reward.
Each auction round is naturally one such group (15-44 responses, same prompt).

Usage:
  python train_from_dataset.py
  python train_from_dataset.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 3 --output outputs/run_01

Requirements:
  pip install torch transformers accelerate datasets pandas tqdm
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = Path(__file__).parent
EXP2_DIR = EXPERIMENTS_DIR / "Experiments-test" / "exp2"
sys.path.insert(0, str(EXP2_DIR))

from grpo_trainer import ExperimentGRPOTrainer  # noqa: E402 (after sys.path)
from utils import group_normalize, ensure_dir   # noqa: E402


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """Load JSONL dataset and build a prompt column."""
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    # Full prompt = system instructions + turn-level context
    df["prompt"] = df["system_prompt"] + "\n\n" + df["user_prompt"]
    # Normalise rewards to [0, 1] (judge scores are 1-10)
    df["reward_raw"] = df["reward"]
    df["reward"] = (df["reward"] - 1.0) / 9.0
    return df


def get_round_groups(df: pd.DataFrame):
    """
    Return list of (round_id, group_df) sorted by round.
    Each group shares an identical prompt; responses and rewards vary.
    """
    groups = []
    for round_id, group_df in df.groupby("round"):
        if len(group_df) < 2:
            continue  # GRPO needs at least 2 responses to normalise
        groups.append((round_id, group_df.reset_index(drop=True)))
    return sorted(groups, key=lambda x: x[0])


# ── Metric tracking ───────────────────────────────────────────────────────────

def print_metrics(epoch: int, round_id: int, metrics: dict):
    parts = [f"epoch={epoch}", f"round={round_id:>3}"]
    for k, v in metrics.items():
        parts.append(f"{k}={v:.4f}")
    print("  " + " | ".join(parts))


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    print(f"Loading dataset from: {args.dataset}")
    df = load_dataset(args.dataset)
    print(f"  {len(df)} rows | {df['round'].nunique()} unique rounds | "
          f"reward range [{df['reward_raw'].min():.0f}, {df['reward_raw'].max():.0f}]")

    groups = get_round_groups(df)
    print(f"  {len(groups)} GRPO groups (rounds with >= 2 responses)")

    ensure_dir(args.output)

    print(f"\nInitialising trainer  model={args.model}")
    trainer = ExperimentGRPOTrainer(
        model_name_or_path=args.model,
        learning_rate=args.lr,
        beta_kl=args.beta_kl,
    )

    all_metrics = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        epoch_loss = 0.0
        epoch_reward = 0.0

        for round_id, group_df in tqdm(groups, desc=f"Epoch {epoch}"):
            metrics = trainer.train_group(
                group_df=group_df,
                max_length=args.max_length,
            )
            metrics["epoch"] = epoch
            metrics["round"] = round_id
            all_metrics.append(metrics)

            epoch_loss += metrics["loss"]
            epoch_reward += metrics["mean_reward"]

        n = len(groups)
        print(f"  avg_loss={epoch_loss/n:.4f} | avg_mean_reward={epoch_reward/n:.4f}")

        ckpt_dir = os.path.join(args.output, f"epoch_{epoch}")
        trainer.save(ckpt_dir)
        print(f"  Saved checkpoint → {ckpt_dir}")

    # Save full metrics log
    metrics_path = os.path.join(args.output, "training_metrics.csv")
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
    print(f"\nMetrics saved → {metrics_path}")

    final_dir = os.path.join(args.output, "final")
    trainer.save(final_dir)
    print(f"Final model saved → {final_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GRPO training on auction_rl_dataset")
    p.add_argument(
        "--dataset",
        default=str(EXPERIMENTS_DIR / "auction_rl_dataset.jsonl"),
        help="Path to auction_rl_dataset.jsonl",
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name or local path (default: Qwen2.5-0.5B-Instruct)",
    )
    p.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    p.add_argument("--beta_kl", type=float, default=0.01, help="KL penalty coefficient")
    p.add_argument("--max_length", type=int, default=768, help="Max token length")
    p.add_argument(
        "--output",
        default=str(EXPERIMENTS_DIR / "outputs" / "grpo_bidder"),
        help="Output directory for checkpoints",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
