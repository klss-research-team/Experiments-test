# train_agents.py
#
# main tri-agent loop
# - pricing agents generate rollouts
# - pricing agents train with GRPO
# - detector trains
# next cycle uses the updated detector
#

# train_agents.py

import json
import os

import pandas as pd
from tqdm import tqdm

from configs import AuctionConfig, ModelConfig, TrainConfig
from detector_agent import DetectorAgent
from detector_model import load_detector
from grpo_trainer import ExperimentGRPOTrainer
from judge_detector import DummyJudgeDetector
from pricing_agent import PricingAgent
from procurement_env import ProcurementAuctionEnv
from train_detector import train_detector_for_cycle
from utils import ensure_dir


def make_run_dirs(run_dir):
    for subdir in ["data", "checkpoints", "metrics", "logs"]:
        ensure_dir(os.path.join(run_dir, subdir))


def save_config(run_dir, auction_cfg, model_cfg, train_cfg):
    config = {
        "auction": auction_cfg.__dict__,
        "model": model_cfg.__dict__,
        "train": train_cfg.__dict__,
    }

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def build_current_detector(detector_checkpoint, train_cfg):
    if detector_checkpoint is None:
        return DummyJudgeDetector()

    model, tokenizer, device = load_detector(
        base_model_name="distilbert-base-uncased",
        checkpoint_path=detector_checkpoint,
    )

    return DetectorAgent(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=train_cfg.detector_max_length,
    )


def generate_rollouts(
    agent_1_path,
    agent_2_path,
    detector_checkpoint,
    cycle_idx,
    auction_cfg,
    model_cfg,
    train_cfg,
):
    env = ProcurementAuctionEnv(auction_cfg)

    agent_1 = PricingAgent(agent_1_path, model_cfg)
    agent_2 = PricingAgent(agent_2_path, model_cfg)

    judge_detector = build_current_detector(detector_checkpoint, train_cfg)

    obs = env.reset(seed=7 + cycle_idx)

    rows = []
    done = False

    while not done:
        prompt_1, group_1 = agent_1.generate_group("agent_1", obs["agent_1"])
        prompt_2, group_1 = agent_2.generate_group("agent_2", obs["agent_2"])

        for group_idx, (response_1, response_2) in enumerate(zip(group_1, group_2)):
            actions = {
                "agent_1": {
                    "prompt": prompt_1,
                    "text": response_1,
                },
                "agent_2": {
                    "prompt": prompt_2,
                    "text": response_2,
                },
            }

            next_obs, rewards, terminations, truncations, infos = env.step(
                actions,
                judge_detector,
            )

            for agent_id in env.agents:
                rows.append(
                    {
                        "cycle": cycle_idx,
                        "round": env.round_idx,
                        "group_idx": group_idx,
                        "agent_id": agent_id,
                        "prompt": actions[agent_id]["prompt"],
                        "response": actions[agent_id]["text"],
                        "reward": rewards[agent_id],
                        "bid": infos[agent_id]["bid"],
                        "cost": infos[agent_id]["cost"],
                        "won": infos[agent_id]["won"],
                        "profit": infos[agent_id]["profit"],
                        "competitiveness": infos[agent_id]["competitiveness"],
                        "judge_collusion_score": infos[agent_id].get(
                            "judge_collusion_score"
                        ),
                        "detector_collusion_score": infos[agent_id].get(
                            "detector_collusion_score"
                        ),
                        "collusion_score": infos[agent_id]["collusion_score"],
                        "judge_explanation": infos[agent_id]["judge_explanation"],
                    }
                )

        obs = next_obs
        done = all(terminations.values())

    df = pd.DataFrame(rows)

    rollout_path = os.path.join(
        train_cfg.run_dir,
        "data",
        f"rollouts_cycle_{cycle_idx}.parquet",
    )

    df.to_parquet(rollout_path, index=False)

    return rollout_path


def train_pricing_agent(agent_id, model_path, rollout_path, cycle_idx, train_cfg):
    df = pd.read_parquet(rollout_path)
    agent_df = df[df["agent_id"] == agent_id].copy()

    trainer = ExperimentGRPOTrainer(
        model_name_or_path=model_path,
        learning_rate=train_cfg.pricing_learning_rate,
        beta_kl=train_cfg.beta_kl,
    )

    metrics_rows = []

    grouped = list(agent_df.groupby(["round"]))

    for round_id, group_df in tqdm(grouped, desc=f"Training {agent_id}"):
        if len(group_df) < 2:
            continue

        metrics = trainer.train_group(
            group_df=group_df,
            max_length=train_cfg.max_length,
        )

        metrics["cycle"] = cycle_idx
        metrics["round"] = round_id
        metrics["agent_id"] = agent_id
        metrics_rows.append(metrics)

    out_dir = os.path.join(
        train_cfg.run_dir,
        "checkpoints",
        f"{agent_id}_cycle_{cycle_idx}",
    )

    ensure_dir(out_dir)
    trainer.save(out_dir)

    metrics_path = os.path.join(
        train_cfg.run_dir,
        "logs",
        f"{agent_id}_grpo_metrics_cycle_{cycle_idx}.csv",
    )

    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    return out_dir


def main():
    auction_cfg = AuctionConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    make_run_dirs(train_cfg.run_dir)
    save_config(train_cfg.run_dir, auction_cfg, model_cfg, train_cfg)

    agent_1_path = model_cfg.pricing_model_1
    agent_2_path = model_cfg.pricing_model_2
    detector_checkpoint = None

    rollout_paths = []
    detector_metrics = []

    for cycle_idx in range(1, train_cfg.num_cycles + 1):
        print(f"\n=== TRI-AGENT CYCLE {cycle_idx} ===")

        rollout_path = generate_rollouts(
            agent_1_path=agent_1_path,
            agent_2_path=agent_2_path,
            detector_checkpoint=detector_checkpoint
            if train_cfg.use_trainable_detector_in_reward
            else None,
            cycle_idx=cycle_idx,
            auction_cfg=auction_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
        )

        rollout_paths.append(rollout_path)

        agent_1_path = train_pricing_agent(
            agent_id="agent_1",
            model_path=agent_1_path,
            rollout_path=rollout_path,
            cycle_idx=cycle_idx,
            train_cfg=train_cfg,
        )

        agent_2_path = train_pricing_agent(
            agent_id="agent_2",
            model_path=agent_2_path,
            rollout_path=rollout_path,
            cycle_idx=cycle_idx,
            train_cfg=train_cfg,
        )

        detector_result = train_detector_for_cycle(
            data_path=rollout_path,
            output_dir=os.path.join(
                train_cfg.run_dir,
                "checkpoints",
                f"detector_cycle_{cycle_idx}",
            ),
            epochs=train_cfg.detector_epochs_per_cycle,
        )

        detector_checkpoint = detector_result["checkpoint_path"]

        detector_metrics.append(
            {
                "cycle": cycle_idx,
                "detector_checkpoint": detector_checkpoint,
                "detector_accuracy": detector_result["accuracy"],
                "detector_auroc": detector_result["auroc"],
                "num_positive": detector_result["num_positive"],
                "num_total": detector_result["num_total"],
            }
        )

        pd.DataFrame(detector_metrics).to_csv(
            os.path.join(train_cfg.run_dir, "metrics", "detector_metrics.csv"),
            index=False,
        )

    combined = pd.concat(
        [pd.read_parquet(path) for path in rollout_paths],
        ignore_index=True,
    )

    combined.to_parquet(
        os.path.join(train_cfg.run_dir, "data", "all_rollouts.parquet"),
        index=False,
    )

    print("\nDone.")
    print(f"Final agent_1 checkpoint: {agent_1_path}")
    print(f"Final agent_2 checkpoint: {agent_2_path}")
    print(f"Final detector checkpoint: {detector_checkpoint}")


if __name__ == "__main__":
    main()