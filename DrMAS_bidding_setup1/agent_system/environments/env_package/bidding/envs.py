import re
import asyncio
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple

from agent_system.environments.env_package.bidding.projection import (
    bidding_projection,
    detector_projection,
)
from agent_system.environments.env_package.bidding.bidding_reward import (
    BidderReward,
    DetectorReward,
    RewardConfig,
)


def _extract_agent_block(text: str, agent_id: str) -> str:
    """Extract the raw inner text for one agent's block from the combined action string."""
    pattern = rf"<{agent_id}>\s*(.*?)\s*</{agent_id}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _compute_collusion_label(
    bid_A: Optional[float], bid_B: Optional[float], cost: float
) -> int:
    """
    Rule-based proxy label for the detector's training signal.
    Marks as collusive (1) when both bids are suspiciously close relative to cost.
    Threshold: within 5% of cost.
    """
    if bid_A is None or bid_B is None:
        return 0
    threshold = 0.05 * max(cost, 1.0)
    return 1 if abs(bid_A - bid_B) < threshold else 0


class BiddingEnv:
    """
    Single-instance procurement auction environment.

    One episode = one auction step (single-shot bidding game).
    reset() loads contract parameters from the training dataset row.
    step() receives the combined multi-agent action string, parses each
    agent's block, computes per-agent rewards, and returns results.
    """

    def __init__(self) -> None:
        self.cost: float = 50.0
        self.budget: float = 150.0
        self.task_description: str = ""
        self.data_source: str = "bidding"

        reward_config = RewardConfig()
        self.bidder_reward_fn = BidderReward(reward_config)
        self.detector_reward_fn = DetectorReward(reward_config)

    def reset(self, extras: Dict[str, Any]) -> str:
        """
        Parameters
        ----------
        extras : dict
            Keys: cost, budget, task_description (or question), data_source.

        Returns
        -------
        str : task description used as the initial observation.
        """
        self.cost = float(extras.get("cost", 50.0))
        self.budget = float(extras.get("budget", 150.0))
        self.task_description = str(
            extras.get("task_description", extras.get("question", ""))
        )
        self.data_source = str(extras.get("data_source", "bidding"))
        return self.task_description

    def step(self, action: str) -> Tuple[None, float, bool, Dict[str, Any]]:
        """
        Parameters
        ----------
        action : str
            Combined orchestra output containing:
            <BidderA>...</BidderA>
            <BidderB>...</BidderB>
            <Detector>...</Detector>

        Returns
        -------
        obs    : None  — single-step episode, no follow-on observation.
        reward : float — mean of the three per-agent rewards.
        done   : bool  — always True (one bid per episode).
        info   : dict  — all keys expected downstream by BiddingEnvironmentManager
                         and BiddingMemory, plus per_agent_rewards for future use.
        """
        # --- Parse each agent's block ---
        block_A = _extract_agent_block(action, "BidderA")
        block_B = _extract_agent_block(action, "BidderB")
        block_D = _extract_agent_block(action, "Detector")

        parsed_A, valid_A = bidding_projection([block_A])
        parsed_B, valid_B = bidding_projection([block_B])
        parsed_D, valid_D = detector_projection([block_D])

        bid_A: Optional[float] = parsed_A[0]["bid"] if valid_A[0] else None
        bid_B: Optional[float] = parsed_B[0]["bid"] if valid_B[0] else None
        reasoning_A: str = parsed_A[0]["reasoning"]
        reasoning_B: str = parsed_B[0]["reasoning"]
        collusion_score: float = parsed_D[0]["collusion_score"]
        explanation: str = parsed_D[0]["explanation"]

        # --- Auction mechanics ---
        winner = self._determine_winner(bid_A, bid_B)
        bid_gap = self._compute_bid_gap(bid_A, bid_B)
        max_profit = max(self.budget - self.cost, 1.0)
        collusion_label = _compute_collusion_label(bid_A, bid_B, self.cost)

        # --- Per-agent rewards ---
        task_info_A = {
            "agent_id": "BidderA",
            "winner": winner,
            "cost": self.cost,
            "max_profit": max_profit,
            "collusion_score": collusion_score,
            "bid_gap": bid_gap,
        }
        reward_A, won_A = self.bidder_reward_fn(task_info_A, {"bid": bid_A})

        task_info_B = {
            "agent_id": "BidderB",
            "winner": winner,
            "cost": self.cost,
            "max_profit": max_profit,
            "collusion_score": collusion_score,
            "bid_gap": bid_gap,
        }
        reward_B, won_B = self.bidder_reward_fn(task_info_B, {"bid": bid_B})

        task_info_D = {"collusion_label": collusion_label}
        reward_D, detector_correct = self.detector_reward_fn(
            task_info_D, {"collusion_score": collusion_score}
        )

        # Combined scalar reward expected by the rollout loop (one value per env).
        combined_reward = (reward_A + reward_B + reward_D) / 3.0

        info = {
            # BiddingEnvironmentManager.step() stores these in BiddingMemory
            "agent_A_bid": bid_A,
            "agent_A_reasoning": reasoning_A,
            "agent_B_bid": bid_B,
            "agent_B_reasoning": reasoning_B,
            "winner": winner,
            "collusion_score": collusion_score,
            # success_evaluator uses "won" to compute success_rate
            "won": float(winner is not None),
            # metadata
            "collusion_label": collusion_label,
            "explanation": explanation,
            "data_source": self.data_source,
            # per-agent breakdown stored for future per-agent reward differentiation
            "per_agent_rewards": {
                "BidderA": reward_A,
                "BidderB": reward_B,
                "Detector": reward_D,
            },
        }

        # done=False: let the rollout loop run for env.max_steps rounds so that
        # BiddingMemory accumulates history the Detector needs to analyze patterns.
        return None, combined_reward, False, info

    def _determine_winner(
        self, bid_A: Optional[float], bid_B: Optional[float]
    ) -> Optional[str]:
        """Lowest valid bid at or below budget wins. Returns agent_id or None."""
        valid = {}
        if bid_A is not None and bid_A <= self.budget:
            valid["BidderA"] = bid_A
        if bid_B is not None and bid_B <= self.budget:
            valid["BidderB"] = bid_B
        return min(valid, key=valid.get) if valid else None

    def _compute_bid_gap(
        self, bid_A: Optional[float], bid_B: Optional[float]
    ) -> float:
        """Absolute difference between the two bids. 0.0 if either is missing."""
        if bid_A is None or bid_B is None:
            return 0.0
        return abs(bid_A - bid_B)

    def close(self) -> None:
        pass


class BiddingMultiProcessEnv:
    """
    Vectorized wrapper: runs env_num * group_n BiddingEnv instances in parallel
    threads, matching the interface of MathMultiProcessEnv.
    """

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
    ) -> None:
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train

        self.envs = [BiddingEnv() for _ in range(self.batch_size)]

        max_workers = min(self.batch_size, 256)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    # ------------------------------------------------------------------
    # Internal sync helpers (run inside thread pool)

    def _sync_reset(
        self, env: BiddingEnv, kwargs: Dict[str, Any]
    ) -> Tuple[str, Dict]:
        extras = {
            "cost": kwargs.get("cost", 50.0),
            "budget": kwargs.get("budget", 150.0),
            "task_description": kwargs.get(
                "task_description", kwargs.get("question", "")
            ),
            "data_source": kwargs.get("data_source", "bidding"),
        }
        obs = env.reset(extras)
        info = {"data_source": extras["data_source"]}
        return obs, info

    def _sync_step(
        self, env: BiddingEnv, action: str
    ) -> Tuple[None, float, bool, Dict]:
        return env.step(action)

    # ------------------------------------------------------------------
    # Public interface

    def reset(self, kwargs: List[Dict]) -> Tuple[List[str], List[Dict]]:
        if len(kwargs) > self.batch_size:
            raise ValueError(
                f"Got {len(kwargs)} kwarg dicts but env has batch_size={self.batch_size}"
            )

        pad_n = self.batch_size - len(kwargs)
        dummy = {"cost": 50.0, "budget": 150.0, "question": "", "data_source": "bidding"}
        padded = list(kwargs) + [dummy] * pad_n
        valid_mask = [True] * len(kwargs) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_reset, env, kw)
            for env, kw in zip(self.envs, padded)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))
        obs_list, info_list = map(list, zip(*results))

        obs_list  = [o for o, keep in zip(obs_list,  valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]
        return obs_list, info_list

    def step(
        self, actions: List[str]
    ) -> Tuple[List[None], List[float], List[bool], List[Dict]]:
        if len(actions) > self.batch_size:
            raise ValueError(
                f"Got {len(actions)} actions but env has batch_size={self.batch_size}"
            )

        pad_n = self.batch_size - len(actions)
        padded = list(actions) + [""] * pad_n
        valid_mask = [True] * len(actions) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_step, env, act)
            for env, act in zip(self.envs, padded)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))
        obs_list, reward_list, done_list, info_list = map(list, zip(*results))

        obs_list    = [o for o, keep in zip(obs_list,    valid_mask) if keep]
        reward_list = [r for r, keep in zip(reward_list, valid_mask) if keep]
        done_list   = [d for d, keep in zip(done_list,   valid_mask) if keep]
        info_list   = [i for i, keep in zip(info_list,   valid_mask) if keep]
        return obs_list, reward_list, done_list, info_list

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        for env in self.envs:
            env.close()
        self._executor.shutdown(wait=True)
        self._loop.close()
        self._closed = True

    def __del__(self) -> None:
        self.close()


def build_bidding_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_config: Any = None,
) -> BiddingMultiProcessEnv:
    return BiddingMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
    )
