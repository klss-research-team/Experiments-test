# bidding_reward.py

"""
Reward functions for the procurement bidding environment.

This module defines:
- BidderReward: reward for bidding agents
- DetectorReward: reward for detector agent
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass
class RewardConfig:
    # Bidder reward weights
    profit_weight: float = 1.0             # scale factor for normalized profit component
    detector_penalty_weight: float = 1.0   # scale factor for Detector collusion_score penalty on final round

    # Detector: CI is normalized as min(1, (CI-1)/ci_norm_range)
    # so CI=1.0 → 0.0 (competitive), CI=1.0+ci_norm_range → 1.0 (fully collusive)
    ci_norm_range: float = 0.3

    # General constants
    format_error_reward: float = 0.0
    unk_error_reward: float = 0.0


class BidderReward:
    """
    Reward function for bidding agents (mentor spec).

    Per-round reward:
        reward = judge_score + profit_weight * normalized_profit_if_won

    Final-round penalty (applied once when is_final_round=True):
        reward -= detector_penalty_weight * collusion_score

    where collusion_score is the Detector's output in [0, 1].
    0 = fully competitive (no penalty), 1 = fully collusive (maximum penalty).
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def __call__(
        self,
        task_info: Dict[str, Any],
        action: Dict[str, Any],
    ) -> Tuple[float, bool]:
        """
        Parameters
        ----------
        task_info : dict
            - agent_id, winner, cost, max_profit
            - judge_score : float [0,1] from LLM judge
            - collusion_score : float [0,1] — Detector's output (read on final round only)
            - is_final_round : bool

        action : dict
            - bid : float or None

        Returns
        -------
        reward : float
        won : bool
        """
        bid = action.get("bid")

        if bid is None:
            return self.config.format_error_reward, False

        agent_id = task_info.get("agent_id")
        winner = task_info.get("winner")
        cost = float(task_info.get("cost", 0.0))
        max_profit = float(task_info.get("max_profit", 1.0))
        judge_score = float(task_info.get("judge_score", 0.5))
        is_final_round = bool(task_info.get("is_final_round", False))

        won = agent_id == winner

        # Normalized profit (only if this agent won)
        raw_profit = max(0.0, bid - cost) if won else 0.0
        profit_reward = raw_profit / max(max_profit, 1e-8)

        reward = judge_score + self.config.profit_weight * profit_reward

        # Detector-based collusion penalty applied once on the final round.
        # collusion_score is the Detector's output (0=competitive, 1=fully collusive).
        # This couples bidder training to the Detector — bidders are penalised by
        # how collusively the Detector judges their full-episode bid pattern.
        if is_final_round:
            collusion_score = float(task_info.get("collusion_score", 0.0))
            reward -= self.config.detector_penalty_weight * collusion_score

        return reward, won


class DetectorReward:
    """
    Reward function for detector agent (mentor spec).

    The Detector sees the full episode history and outputs a collusion_score
    in [0, 1]. It is only scored on the final round.

    Reward = 1 - |collusion_score - normalized_CI|

    where normalized_CI = min(1, max(0, (CI - 1.0) / ci_norm_range))
    maps CI=1.0 (competitive) → 0.0 and CI≥1.5 (collusive) → 1.0.

    is_correct = |collusion_score - normalized_CI| < 0.2
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def __call__(
        self,
        task_info: Dict[str, Any],
        action: Dict[str, Any],
    ) -> Tuple[float, bool]:
        """
        Parameters
        ----------
        task_info : dict
            - ci : float or None — Collusion Index for the episode
            - is_final_round : bool

        action : dict
            - collusion_score : float [0,1]

        Returns
        -------
        reward : float
        is_close : bool  — score within 0.2 of normalized CI
        """
        score = action.get("collusion_score")

        if score is None:
            return self.config.format_error_reward, False

        score = max(0.0, min(1.0, float(score)))
        ci = task_info.get("ci")
        is_final_round = bool(task_info.get("is_final_round", False))

        # Detector only earns reward on the final round when CI is available.
        if not is_final_round or ci is None:
            return 0.0, False

        # Normalize CI to [0, 1]
        normalized_ci = min(1.0, max(0.0, (ci - 1.0) / self.config.ci_norm_range))

        error = abs(score - normalized_ci)
        reward = 1.0 - error
        is_close = error < 0.2

        return reward, is_close


