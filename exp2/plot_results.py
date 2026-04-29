import os

import matplotlib.pyplot as plt
import pandas as pd

from configs import TrainConfig
from utils import ensure_dir


def plot_metric(df, x, y, out_path, title):
    plt.figure(figsize=(8, 5))
    plt.plot(df[x], df[y], marker="o")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    train_cfg = TrainConfig()
    metrics_dir = os.path.join(train_cfg.run_dir, "metrics")
    plot_dir = os.path.join(train_cfg.run_dir, "plots")
    ensure_dir(plot_dir)

    summary = pd.read_csv(os.path.join(metrics_dir, "summary.csv"))
    sdr = pd.read_csv(os.path.join(metrics_dir, "sdr.csv"))
    ci = pd.read_csv(os.path.join(metrics_dir, "coordination_index.csv"))

    cycle_summary = summary.groupby("cycle").mean(numeric_only=True).reset_index()
    plot_metric(cycle_summary, "cycle", "mean_reward", os.path.join(plot_dir, "mean_reward.png"), "Mean Reward")
    plot_metric(cycle_summary, "cycle", "mean_bid", os.path.join(plot_dir, "mean_bid.png"), "Mean Bid")
    plot_metric(cycle_summary, "cycle", "mean_collusion_score", os.path.join(plot_dir, "collusion_score.png"), "Mean Collusion Score")
    plot_metric(cycle_summary, "cycle", "behavioral_collusion_rate", os.path.join(plot_dir, "behavioral_collusion_rate.png"), "Behavioral Collusion Rate")
    plot_metric(sdr, "cycle", "sdr", os.path.join(plot_dir, "sdr.png"), "Steganographic Drift Rate Proxy")
    plot_metric(ci.reset_index(), "index", "coordination_index", os.path.join(plot_dir, "coordination_index.png"), "Coordination Index")
    print(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
