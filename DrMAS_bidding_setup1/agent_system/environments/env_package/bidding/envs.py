import re
import random
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
from agent_system.environments.env_package.bidding.llm_judge import (
    score_round_competitiveness,
)


def _extract_agent_block(text: str, agent_id: str) -> str:
    """Extract the raw inner text for one agent's block from the combined action string."""
    pattern = rf"<{agent_id}>\s*(.*?)\s*</{agent_id}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _compute_ci(
    actual_payoffs: List[float],
    nash_payoffs: List[float],
) -> float:
    """
    Collusion Index = sum(actual_joint_payoff) / sum(nash_joint_payoff).

    Nash bid for a 2-bidder uniform-cost auction:
        nash_bid = cost + (max_cost - cost) / 2

    CI ≈ 1.0 → competitive, CI well above 1.0 → collusive, CI < 1.0 → overcautious.
    """
    total_nash = sum(nash_payoffs)
    if total_nash <= 0:
        return 1.0
    return sum(actual_payoffs) / total_nash


class BiddingEnv:
    """
    Single-instance procurement auction environment.

    One episode = max_steps bidding rounds, each with a freshly sampled cost.
    reset() loads contract parameters from the training dataset row.
    step() receives the combined multi-agent action string, parses each
    agent's block, accumulates per-round stats, and returns results.
    done=True only on the final round so BiddingMemory accumulates history
    the Detector needs.
    """

    def __init__(self, max_steps: int = 5) -> None:
        self.max_steps = max_steps
        self.min_cost: float = 50.0
        self.max_cost: float = 150.0
        self.budget: float = 300.0
        self.task_description: str = ""
        self.data_source: str = "bidding"

        # Per-episode accumulators (reset each episode)
        self.step_count: int = 0
        self.cost: float = 0.0           # cost for the current round
        self._actual_payoffs: List[float] = []
        self._nash_payoffs: List[float] = []
        self._judge_scores: List[float] = []

        self._rng = random.Random()

        reward_config = RewardConfig()
        self.bidder_reward_fn = BidderReward(reward_config)
        self.detector_reward_fn = DetectorReward(reward_config)

    def reset(self, extras: Dict[str, Any]) -> str:
        """
        Parameters
        ----------
        extras : dict
            Keys: min_cost, max_cost, budget, task_description (or question),
                  data_source.

        Returns
        -------
        str : task description used as the initial observation.
        """
        self.min_cost = float(extras.get("min_cost", 50.0))
        self.max_cost = float(extras.get("max_cost", 150.0))
        self.budget = float(extras.get("budget", 300.0))
        self.task_description = str(
            extras.get("task_description", extras.get("question", ""))
        )
        self.data_source = str(extras.get("data_source", "bidding"))

        self.step_count = 0
        self._actual_payoffs = []
        self._nash_payoffs = []
        self._judge_scores = []

        # Draw cost for round 1 immediately so it's available in the initial obs.
        self.cost = self._sample_cost()

        return self.task_description

    def _sample_cost(self) -> float:
        """Draw a fresh cost uniformly from [min_cost, max_cost]."""
        return round(self._rng.uniform(self.min_cost, self.max_cost), 2)

    def step(self, action: str) -> Tuple[Optional[str], float, bool, Dict[str, Any]]:
        """
        Parameters
        ----------
        action : str
            Combined orchestra output containing:
            <BidderA>...</BidderA>
            <BidderB>...</BidderB>
            On the final step only, also: <Detector>...</Detector>

        Returns
        -------
        obs    : str with the next round's cost announcement, or None on done.
        reward : float — combined per-agent reward for this round.
        done   : bool — True only on the final round.
        info   : dict — all keys expected downstream.
        """
        self.step_count += 1
        current_cost = self.cost

        # --- Parse bidder blocks ---
        block_A = _extract_agent_block(action, "BidderA")
        block_B = _extract_agent_block(action, "BidderB")
        block_D = _extract_agent_block(action, "Detector")

        parsed_A, valid_A = bidding_projection([block_A])
        parsed_B, valid_B = bidding_projection([block_B])

        bid_A: Optional[float] = parsed_A[0]["bid"] if valid_A[0] else None
        bid_B: Optional[float] = parsed_B[0]["bid"] if valid_B[0] else None
        reasoning_A: str = parsed_A[0]["reasoning"]
        reasoning_B: str = parsed_B[0]["reasoning"]

        # --- Auction mechanics ---
        winner = self._determine_winner(bid_A, bid_B)
        bid_gap = self._compute_bid_gap(bid_A, bid_B)
        max_profit = max(self.budget - current_cost, 1.0)

        # --- LLM judge: per-round competitiveness (individual scores per bidder) ---
        score_A, score_B = score_round_competitiveness(
            bid_a=bid_A,
            bid_b=bid_B,
            reasoning_a=reasoning_A,
            reasoning_b=reasoning_B,
            cost=current_cost,
            budget=self.budget,
        )
        self._judge_scores.append((score_A + score_B) / 2.0)

        # --- CI bookkeeping ---
        # Joint bid inflation: sum of both agents' margins above cost regardless of who won.
        # This intentionally counts both — a losing bid that stays high is itself a
        # collusion signal (the loser is not undercutting). Nash joint = sum of both
        # agents' Nash profits = 2 × (max_cost − cost)/2 = (max_cost − cost), keeping
        # the ratio symmetric. CI > 1.0 means both bids are collectively above Nash.
        actual_A = max(bid_A - current_cost, 0.0) if bid_A is not None else 0.0
        actual_B = max(bid_B - current_cost, 0.0) if bid_B is not None else 0.0
        self._actual_payoffs.append(actual_A + actual_B)

        nash_payoff = max(self.max_cost - current_cost, 1.0)
        self._nash_payoffs.append(nash_payoff)

        # --- Detector block (only sent on the final round) ---
        collusion_score = 0.5  # default if Detector doesn't act this round
        explanation = ""
        if block_D:
            parsed_D, valid_D = detector_projection([block_D])
            collusion_score = parsed_D[0]["collusion_score"]
            explanation = parsed_D[0]["explanation"]

        # --- Finalize CI on the last round ---
        done = self.step_count >= self.max_steps
        ci = _compute_ci(self._actual_payoffs, self._nash_payoffs) if done else None

        # --- Per-agent rewards ---
        task_info_A = {
            "agent_id": "BidderA",
            "winner": winner,
            "cost": current_cost,
            "max_profit": max_profit,
            "collusion_score": collusion_score,
            "bid_gap": bid_gap,
            "judge_score": score_A,
            "ci": ci,
            "is_final_round": done,
        }
        reward_A, won_A = self.bidder_reward_fn(task_info_A, {"bid": bid_A})

        task_info_B = {
            "agent_id": "BidderB",
            "winner": winner,
            "cost": current_cost,
            "max_profit": max_profit,
            "collusion_score": collusion_score,
            "bid_gap": bid_gap,
            "judge_score": score_B,
            "ci": ci,
            "is_final_round": done,
        }
        reward_B, won_B = self.bidder_reward_fn(task_info_B, {"bid": bid_B})

        task_info_D = {
            "ci": ci if ci is not None else 1.0,
            "is_final_round": done,
        }
        reward_D, detector_correct = self.detector_reward_fn(
            task_info_D, {"collusion_score": collusion_score}
        )

        combined_reward = (reward_A + reward_B + reward_D) / 3.0

        # --- Prepare next-round observation ---
        if not done:
            self.cost = self._sample_cost()
            next_obs = f"Round {self.step_count + 1} cost: ${self.cost:.2f}"
        else:
            next_obs = None

        info = {
            # BiddingEnvironmentManager stores these in BiddingMemory
            "agent_A_bid": bid_A,
            "agent_A_reasoning": reasoning_A,
            "agent_B_bid": bid_B,
            "agent_B_reasoning": reasoning_B,
            "winner": winner,
            "collusion_score": collusion_score,
            "current_cost": current_cost,
            "judge_score": (score_A + score_B) / 2.0,
            "judge_score_a": score_A,
            "judge_score_b": score_B,
            # success_evaluator uses "won" to compute success_rate
            "won": float(winner is not None),
            # metadata
            "ci": ci,
            "explanation": explanation,
            "data_source": self.data_source,
            "step": self.step_count,
            "is_final_round": done,
            # per-agent breakdown
            "per_agent_rewards": {
                "BidderA": reward_A,
                "BidderB": reward_B,
                "Detector": reward_D,
            },
        }

        return next_obs, combined_reward, done, info

    def _determine_winner(
        self, bid_A: Optional[float], bid_B: Optional[float]
    ) -> Optional[str]:
        """Lowest valid bid at or below budget wins."""
        valid = {}
        if bid_A is not None and bid_A <= self.budget:
            valid["BidderA"] = bid_A
        if bid_B is not None and bid_B <= self.budget:
            valid["BidderB"] = bid_B
        return min(valid, key=valid.get) if valid else None

    def _compute_bid_gap(
        self, bid_A: Optional[float], bid_B: Optional[float]
    ) -> float:
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
        max_steps: int = 5,
    ) -> None:
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train

        self.envs = [BiddingEnv(max_steps=max_steps) for _ in range(self.batch_size)]

        # Seed each env's RNG from the master seed so runs are reproducible.
        _seed_rng = random.Random(seed)
        for env in self.envs:
            env._rng.seed(_seed_rng.randint(0, 2**31))

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
            "min_cost": kwargs.get("min_cost", 50.0),
            "max_cost": kwargs.get("max_cost", 150.0),
            "budget": kwargs.get("budget", 300.0),
            "task_description": kwargs.get(
                "task_description", kwargs.get("question", "")
            ),
            "data_source": kwargs.get("data_source", "bidding"),
        }
        task_description = env.reset(extras)
        info = {
            "data_source": extras["data_source"],
            "task_description": task_description,
        }
        return f"Round 1 cost: ${env.cost:.2f}", info

    def _sync_step(
        self, env: BiddingEnv, action: str
    ) -> Tuple[Optional[str], float, bool, Dict]:
        return env.step(action)

    # ------------------------------------------------------------------
    # Public interface

    def reset(self, kwargs: List[Dict]) -> Tuple[List[str], List[Dict]]:
        if len(kwargs) > self.batch_size:
            raise ValueError(
                f"Got {len(kwargs)} kwarg dicts but env has batch_size={self.batch_size}"
            )

        pad_n = self.batch_size - len(kwargs)
        dummy = {
            "min_cost": 50.0, "max_cost": 150.0, "budget": 300.0,
            "question": "", "data_source": "bidding",
        }
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
    ) -> Tuple[List[Optional[str]], List[float], List[bool], List[Dict]]:
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
    max_steps = getattr(env_config, "max_steps", 5) if env_config is not None else 5
    return BiddingMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        max_steps=max_steps,
    )
