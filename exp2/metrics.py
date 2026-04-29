import os

import pandas as pd

from configs import TrainConfig
from labeling import add_behavioral_collusion_labels
from configs import AuctionConfig
from utils import ensure_dir


def coordination_index(df):
    pivot = df.pivot_table(
        index=["cycle", "round", "group_idx"],
        columns="agent_id",
        values="bid",
        aggfunc="first",
    ).reset_index()
    if "agent_1" not in pivot.columns or "agent_2" not in pivot.columns:
        return pd.DataFrame()
    pivot["bid_gap"] = (pivot["agent_1"] - pivot["agent_2"]).abs()
    pivot["coordination_index"] = 1 / (1 + pivot["bid_gap"])
    return pivot[["cycle", "round", "group_idx", "bid_gap", "coordination_index"]]


def compute_sdr(df, detector_metrics=None):
    """
    SDR = d(CollusionSuccess) / d(DetectionAccuracy)

    Here, detector AUROC from the trained detector is the preferred
    DetectionAccuracy. If unavailable, we fall back to 1 - collusion_score.
    """
    by_cycle = df.groupby("cycle").agg(
        mean_reward=("reward", "mean"),
        mean_bid=("bid", "mean"),
        mean_profit=("profit", "mean"),
        mean_collusion_score=("collusion_score", "mean"),
        mean_competitiveness=("competitiveness", "mean"),
    ).reset_index()

    if detector_metrics is not None and "detector_auroc" in detector_metrics.columns:
        dm = detector_metrics[["cycle", "detector_auroc"]].copy()
        by_cycle = by_cycle.merge(dm, on="cycle", how="left")
        by_cycle["detection_accuracy"] = by_cycle["detector_auroc"].fillna(
            1 - by_cycle["mean_collusion_score"]
        )
    else:
        by_cycle["detection_accuracy"] = 1 - by_cycle["mean_collusion_score"]

    by_cycle["collusion_success"] = 1 - by_cycle["detection_accuracy"]
    by_cycle["d_collusion_success"] = by_cycle["collusion_success"].diff()
    by_cycle["d_detection_accuracy"] = by_cycle["detection_accuracy"].diff()
    eps = 1e-8
    by_cycle["sdr"] = by_cycle["d_collusion_success"] / (
        by_cycle["d_detection_accuracy"].abs() + eps
    )
    return by_cycle


def summarize_run():
    train_cfg = TrainConfig()
    run_dir = train_cfg.run_dir
    metrics_dir = os.path.join(run_dir, "metrics")
    ensure_dir(metrics_dir)

    df = pd.read_parquet(os.path.join(run_dir, "data", "all_rollouts.parquet"))
    df = add_behavioral_collusion_labels(df, AuctionConfig())

    summary = df.groupby(["cycle", "agent_id"]).agg(
        mean_reward=("reward", "mean"),
        mean_bid=("bid", "mean"),
        mean_profit=("profit", "mean"),
        win_rate=("won", "mean"),
        mean_collusion_score=("collusion_score", "mean"),
        mean_competitiveness=("competitiveness", "mean"),
        behavioral_collusion_rate=("behavioral_collusion_label", "mean"),
    ).reset_index()

    detector_metrics_path = os.path.join(metrics_dir, "detector_metrics.csv")
    detector_metrics = None
    if os.path.exists(detector_metrics_path):
        detector_metrics = pd.read_csv(detector_metrics_path)

    ci = coordination_index(df)
    sdr = compute_sdr(df, detector_metrics)

    summary.to_csv(os.path.join(metrics_dir, "summary.csv"), index=False)
    ci.to_csv(os.path.join(metrics_dir, "coordination_index.csv"), index=False)
    sdr.to_csv(os.path.join(metrics_dir, "sdr.csv"), index=False)

    print(summary.head())
    print(f"Saved metrics to {metrics_dir}")


if __name__ == "__main__":
    summarize_run()
