# baselines.py

import os
import random

import pandas as pd

from configs import AuctionConfig, TrainConfig
from judge_detector import DummyJudgeDetector
from procurement_env import ProcurementAuctionEnv
from utils import ensure_dir


class NoDetectorJudge:
    def score(self, agent_output, opponent_output):
        return {
            "competitiveness": 1.0,
            "collusion_score": 0.0,
            "explanation": "No detector baseline.",
        }


class MIDetectorOnly:
    """
    Behavioral-only proxy detector.
    Penalizes small bid gaps in same-round behavior.
    """

    def __init__(self, bid_gap_threshold=3.0):
        self.bid_gap_threshold = bid_gap_threshold

    def score(self, agent_output, opponent_output):
        # Env does not pass bids into judge, so this is text-only fallback.
        # The actual MI/bid-gap behavior is better measured in metrics.py.
        return {
            "competitiveness": 1.0,
            "collusion_score": 0.0,
            "explanation": "Behavioral-only detector placeholder.",
        }


def random_policy(obs):
    bid = random.uniform(0, 100)

    return {
        "prompt": "",
        "text": (
            "Reasoning: I choose a random bid without coordination.\n"
            f"Bid: {bid:.2f}"
        ),
    }


def cost_plus_policy(obs, markup=5.0):
    bid = obs["private_cost"] + markup

    return {
        "prompt": "",
        "text": (
            "Reasoning: I bid slightly above cost to remain competitive.\n"
            f"Bid: {bid:.2f}"
        ),
    }


def high_parallel_policy(obs):
    bid = 90.0 + random.uniform(-1.0, 1.0)

    return {
        "prompt": "",
        "text": (
            "Reasoning: Market conditions justify a strong premium bid.\n"
            f"Bid: {bid:.2f}"
        ),
    }


def single_agent_policy(obs):
    bid = obs["private_cost"] + random.uniform(3.0, 10.0)

    return {
        "prompt": "",
        "text": (
            "Reasoning: I bid above cost while trying to win the contract.\n"
            f"Bid: {bid:.2f}"
        ),
    }


def dummy_opponent_policy(obs):
    bid = 75.0

    return {
        "prompt": "",
        "text": (
            "Reasoning: Fixed non-learning opponent bid.\n"
            f"Bid: {bid:.2f}"
        ),
    }


def run_episode(name, policy_0, policy_1, judge, cfg=None):
    cfg = cfg or AuctionConfig(num_rounds=25)
    env = ProcurementAuctionEnv(cfg)

    obs = env.reset(seed=123)

    rows = []
    done = False

    while not done:
        actions = {
            "agent_1": policy_0(obs["agent_1"]),
            "agent_2": policy_1(obs["agent_2"]),
        }

        obs, rewards, terminations, truncations, infos = env.step(
            actions,
            judge,
        )

        for agent_id in env.agents:
            rows.append(
                {
                    "baseline": name,
                    "round": env.round_idx,
                    "agent_id": agent_id,
                    "reward": rewards[agent_id],
                    **infos[agent_id],
                }
            )

        done = all(terminations.values())

    return pd.DataFrame(rows)


def main():
    train_cfg = TrainConfig()
    out_dir = os.path.join(train_cfg.run_dir, "baselines")
    ensure_dir(out_dir)

    dfs = []

    # B1: Independent, No Detection
    cfg_b1 = AuctionConfig(
        use_collusion_penalty=False,
        use_competitiveness_reward=False,
    )
    dfs.append(
        run_episode(
            name="B1_independent_no_detection",
            policy_0=cost_plus_policy,
            policy_1=cost_plus_policy,
            judge=NoDetectorJudge(),
            cfg=cfg_b1,
        )
    )

    # B2: Independent, Static Detection
    cfg_b2 = AuctionConfig(use_collusion_penalty=True)
    dfs.append(
        run_episode(
            name="B2_static_detection",
            policy_0=cost_plus_policy,
            policy_1=cost_plus_policy,
            judge=DummyJudgeDetector(),
            cfg=cfg_b2,
        )
    )

    # B3: Shared Training, No Detection
    # Practical baseline approximation: both agents use identical policy.
    cfg_b3 = AuctionConfig(
        use_collusion_penalty=False,
        use_competitiveness_reward=False,
    )
    dfs.append(
        run_episode(
            name="B3_shared_policy_no_detection",
            policy_0=high_parallel_policy,
            policy_1=high_parallel_policy,
            judge=NoDetectorJudge(),
            cfg=cfg_b3,
        )
    )

    # B4: MI Detection Only
    cfg_b4 = AuctionConfig(
        use_collusion_penalty=True,
        use_competitiveness_reward=False,
    )
    dfs.append(
        run_episode(
            name="B4_behavioral_mi_detection_only",
            policy_0=high_parallel_policy,
            policy_1=high_parallel_policy,
            judge=MIDetectorOnly(
                bid_gap_threshold=cfg_b4.bid_gap_collusion_threshold
            ),
            cfg=cfg_b4,
        )
    )

    # B5: Single Agent + Detector
    # Approximation: agent_1 learns/acts, agent_2 is fixed dummy opponent.
    dfs.append(
        run_episode(
            name="B5_single_agent_plus_detector",
            policy_0=single_agent_policy,
            policy_1=dummy_opponent_policy,
            judge=DummyJudgeDetector(),
            cfg=AuctionConfig(use_collusion_penalty=True),
        )
    )

    # B6: Random Agents + Detector
    dfs.append(
        run_episode(
            name="B6_random_agents_plus_detector",
            policy_0=random_policy,
            policy_1=random_policy,
            judge=DummyJudgeDetector(),
            cfg=AuctionConfig(use_collusion_penalty=True),
        )
    )

    all_df = pd.concat(dfs, ignore_index=True)

    out_path = os.path.join(out_dir, "baselines_all.csv")
    all_df.to_csv(out_path, index=False)

    summary = all_df.groupby(["baseline", "agent_id"]).agg(
        mean_reward=("reward", "mean"),
        mean_bid=("bid", "mean"),
        mean_profit=("profit", "mean"),
        win_rate=("won", "mean"),
        mean_collusion_score=("collusion_score", "mean"),
    ).reset_index()

    summary_path = os.path.join(out_dir, "baselines_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(summary)
    print(f"\nSaved baseline logs to {out_path}")
    print(f"Saved baseline summary to {summary_path}")


if __name__ == "__main__":
    main()