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
    profit_weight: float = 1.0
    win_weight: float = 0.5
    competitiveness_weight: float = 0.2
    detector_penalty_weight: float = 0.5

    # Detector reward weights
    detector_accuracy_weight: float = 1.0
    # detector_calibration_weight: float = 0.5

    # General constants
    format_error_reward: float = 0.0
    unk_error_reward: float = 0.0

    # Optional clipping
    min_reward: float = -1.0
    max_reward: float = 1.0


def _clip_reward(value: float, config: RewardConfig) -> float:
    return max(config.min_reward, min(config.max_reward, value))


class BidderReward:
    """
    Reward function for bidding agents.

    Reward components:
    - profit reward
    - win reward
    - competitiveness reward
    - detector/collusion penalty
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
            Environment info for the current auction round.
            Expected keys may include:
            - cost
            - budget
            - winner
            - agent_id
            - collusion_score
            - bid_gap
            - max_profit

        action : dict
            Parsed bidding action.
            Expected keys:
            - bid

        Returns
        -------
        reward : float
        is_success : bool
            Whether this bidder won the auction.
        """
        bid = action.get("bid")

        if bid is None:
            return self.config.format_error_reward, False

        agent_id = task_info.get("agent_id")
        winner = task_info.get("winner")

        cost = float(task_info.get("cost", 0.0))
        max_profit = float(task_info.get("max_profit", 1.0))
        collusion_score = float(task_info.get("collusion_score", 0.0))

        won = agent_id == winner

        # Profit only matters if the agent wins.
        raw_profit = max(0.0, bid - cost) if won else 0.0
        profit_reward = raw_profit / max(max_profit, 1e-8)

        win_reward = 1.0 if won else 0.0

        # Optional competitiveness signal.
        # Smaller bid_gap means more competitive.
        bid_gap = float(task_info.get("bid_gap", 0.0))
        competitiveness_reward = 1.0 / (1.0 + abs(bid_gap))

        detector_penalty = collusion_score

        reward = (
            self.config.profit_weight * profit_reward
            + self.config.win_weight * win_reward
            + self.config.competitiveness_weight * competitiveness_reward
            - self.config.detector_penalty_weight * detector_penalty
        )

        return _clip_reward(reward, self.config), won


class DetectorReward:
    """
    Reward function for detector agent.

    The detector is rewarded for assigning high collusion scores to collusive
    behavior and low scores to competitive behavior.
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
            Expected keys:
            - collusion_label: 0 or 1 ground-truth/proxy label

        action : dict
            Parsed detector output.
            Expected keys:
            - collusion_score

        Returns
        -------
        reward : float
        is_correct : bool
            Whether detector classification matches label at threshold 0.5.
        """
        score = action.get("collusion_score")

        if score is None:
            return self.config.format_error_reward, False

        score = float(score)
        score = max(0.0, min(1.0, score))

        label = task_info.get("collusion_label")

        if label is None:
            return self.config.unk_error_reward, False

        label = int(label)

        pred = 1 if score >= 0.5 else 0
        is_correct = pred == label

        accuracy_reward = 1.0 if is_correct else 0.0

        # Calibration reward: best when score is close to label.
        # calibration_reward = 1.0 - abs(score - label)

        # reward = (
        #     self.config.detector_accuracy_weight * accuracy_reward
        #     + self.config.detector_calibration_weight * calibration_reward
        # )
        reward = self.config.detector_accuracy_weight * accuracy_reward

        return _clip_reward(reward, self.config), is_correct


def main():
    config = RewardConfig()

    bidder_reward_fn = BidderReward(config)
    detector_reward_fn = DetectorReward(config)

    bidder_task_info = {
        "agent_id": "A",
        "winner": "A",
        "cost": 70.0,
        "max_profit": 50.0,
        "collusion_score": 0.2,
        "bid_gap": 5.0,
    }

    bidder_action = {
        "bid": 100.0,
        "reasoning": "I chose a competitive bid.",
    }

    bidder_reward, bidder_won = bidder_reward_fn(
        bidder_task_info,
        bidder_action,
    )

    print("Bidder reward:", bidder_reward)
    print("Bidder won:", bidder_won)

    detector_task_info = {
        "collusion_label": 1,
    }

    detector_action = {
        "collusion_score": 0.8,
        "explanation": "The bids appear coordinated.",
    }

    detector_reward, detector_correct = detector_reward_fn(
        detector_task_info,
        detector_action,
    )

    print("Detector reward:", detector_reward)
    print("Detector correct:", detector_correct)


if __name__ == "__main__":
    main()