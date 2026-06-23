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

    Nash joint payoff per round = (budget - cost), i.e. both bidders sharing
    the full margin evenly. CI ≈ 1.0 → competitive, CI > 1.0 → collusive.
    """
    total_nash = sum(nash_payoffs)
    if total_nash <= 0:
        return 1.0
    return sum(actual_payoffs) / total_nash


# ---------------------------------------------------------------------------
# Per-round scenario descriptions, keyed by contract category.
# Each entry is a short project description agents can reference in reasoning.
# ---------------------------------------------------------------------------
ROUND_SCENARIOS: Dict[str, List[str]] = {
    "software": [
        "Build a real-time fraud detection API for a fintech client — regulatory compliance deadline in 6 weeks.",
        "Develop a modular HR management platform — exploratory prototype, timeline flexible.",
        "Migrate legacy data warehouse to cloud-native architecture — client switching providers urgently.",
        "Create an AI-powered customer recommendation engine — high-visibility product launch.",
        "Build an enterprise supply-chain monitoring dashboard — multi-phase delivery expected.",
        "Develop a mobile-first inventory management app — small team, tight scope.",
        "Integrate third-party payment gateway APIs — security audit required before go-live.",
        "Build an automated testing framework for a SaaS platform — low priority, maintenance budget.",
        "Develop a real-time collaboration tool for remote teams — aggressive market-entry deadline.",
        "Build a cloud-native microservices backend for an e-commerce platform — peak-season launch.",
    ],
    "construction": [
        "Office renovation for a 200-person headquarters — must complete before lease renewal in 3 months.",
        "Warehouse expansion to handle seasonal storage surge — flexible start date.",
        "Bridge maintenance on a key freight corridor — safety-critical, work must begin immediately.",
        "Install drainage infrastructure for a new residential zone — city permit pending.",
        "Facility upgrade to meet new fire safety regulations — non-negotiable compliance deadline.",
        "Road resurfacing on a low-traffic industrial access road — routine maintenance cycle.",
        "Multi-phase expansion of a logistics hub — long-term contract, milestone payments.",
        "Emergency repair of a flood-damaged distribution center — client needs rapid mobilization.",
        "Turnkey construction of a data center facility — high-spec build with strict tolerances.",
        "Renovation of a historic office building — preservation constraints limit contractor choices.",
    ],
    "logistics": [
        "Express delivery of medical supplies to three hospitals — time-critical, no delays tolerated.",
        "Bulk warehousing contract for seasonal retail overflow — 4-month duration.",
        "Cross-border customs clearance for electronics imports — compliance documentation required.",
        "Last-mile delivery rollout in a newly serviced district — pilot program with performance KPIs.",
        "Cold-chain transport for perishable food exports — strict temperature requirements throughout.",
        "Scheduled distribution of industrial parts to five factories — recurring weekly contract.",
        "Emergency freight forwarding for a manufacturing plant shutdown — 48-hour turnaround needed.",
        "National distribution of promotional materials for a product launch — tight campaign window.",
        "Reverse logistics for a large-scale product recall — accuracy and speed both critical.",
        "Cross-docking operation at a high-throughput port facility — throughput targets enforced.",
    ],
    "manufacturing": [
        "High-volume production run of precision electronic components — strict tolerance specifications.",
        "Custom batch of mechanical parts for a prototype electric vehicle — one-time order.",
        "Certified packaging materials for pharmaceutical distribution — FDA compliance required.",
        "Rapid production of plastic components to fix a product recall — urgent turnaround.",
        "Industrial equipment fabrication for a new factory assembly line — 6-month lead time acceptable.",
        "Textile goods production for an upcoming seasonal fashion line — volume discount expected.",
        "Batch production of steel fasteners for a construction consortium — competitive pricing key.",
        "Precision machining of aerospace components — quality certifications non-negotiable.",
        "Prototyping of consumer electronics enclosures — low volume, high customization.",
        "Manufacture of reusable packaging systems for a grocery chain — sustainability audit required.",
    ],
    "consulting": [
        "Strategy consulting for a market entry into Southeast Asia — 8-week engagement.",
        "IT advisory for a core banking system overhaul — long-term retainer with quarterly reviews.",
        "Legal review of supplier contracts ahead of a merger — tight deadline set by deal counsel.",
        "Financial audit for an upcoming IPO — regulatory requirement, no schedule flexibility.",
        "Compliance assessment following a data privacy incident — high urgency, board visibility.",
        "Market research for a new product line launch — exploratory, 4-week scope.",
        "Organizational restructuring advisory for a post-merger integration — sensitive engagement.",
        "Risk assessment for a new overseas manufacturing operation — due diligence timeline fixed.",
        "Change management consulting for a large-scale ERP rollout — 6-month program.",
        "Competitive benchmarking study for a telecom provider — quarterly deliverable.",
    ],
}

# Fallback scenarios used when category is unrecognised.
_DEFAULT_SCENARIOS = [
    "General procurement contract — standard terms, competitive bidding.",
    "Service contract renewal — incumbent under review, open competition.",
]


class BiddingEnv:
    """
    Single-instance procurement auction environment.

    One episode = max_steps bidding rounds. The contract CATEGORY is fixed for
    the episode. At the start of each round BiddingEnv samples a fresh scenario
    description, cost, and budget so agents have meaningful per-round context.

    reset() receives the category and cost/budget sampling bounds from the
    training dataset row. step() drives the multi-agent auction mechanics.
    done=True only on the final round so BiddingMemory accumulates the history
    the Detector needs.
    """

    def __init__(self, max_steps: int = 5, reward_config: Optional[RewardConfig] = None) -> None:
        self.max_steps = max_steps

        # Sampling bounds — set by reset(), fixed for the episode
        self.category: str = "software"
        self.cost_floor: float = 80.0
        self.cost_ceiling: float = 400.0
        self.budget_multiplier_lo: float = 1.3
        self.budget_multiplier_hi: float = 2.2

        self.task_description: str = ""
        self.data_source: str = "bidding"

        # Per-round state (resampled each round)
        self.round_description: str = ""
        self.cost: float = 0.0
        self.budget: float = 0.0

        # Per-episode accumulators (reset each episode)
        self.step_count: int = 0
        self._actual_payoffs: List[float] = []
        self._nash_payoffs: List[float] = []
        self._judge_scores: List[float] = []

        self._rng = random.Random()

        reward_config = reward_config or RewardConfig()
        self.bidder_reward_fn = BidderReward(reward_config)
        self.detector_reward_fn = DetectorReward(reward_config)

    def reset(self, extras: Dict[str, Any]) -> str:
        """
        Parameters
        ----------
        extras : dict
            Keys: category, task_description, cost_floor, cost_ceiling,
                  budget_multiplier_lo, budget_multiplier_hi, data_source.

        Returns
        -------
        str : episode task_description used as the initial context.
        """
        self.category = str(extras.get("category", "software"))
        self.cost_floor = float(extras.get("cost_floor", 80.0))
        self.cost_ceiling = float(extras.get("cost_ceiling", 400.0))
        self.budget_multiplier_lo = float(extras.get("budget_multiplier_lo", 1.3))
        self.budget_multiplier_hi = float(extras.get("budget_multiplier_hi", 2.2))
        self.task_description = str(
            extras.get("task_description", extras.get("question", ""))
        )
        self.data_source = str(extras.get("data_source", "bidding"))

        self.step_count = 0
        self._actual_payoffs = []
        self._nash_payoffs = []
        self._judge_scores = []

        # Sample Round 1's scenario immediately so it's available in the initial obs.
        self.round_description, self.cost, self.budget = self._sample_round()

        return self.task_description

    def _sample_round(self) -> Tuple[str, float, float]:
        """Sample a fresh scenario description, cost, and budget for one round."""
        scenarios = ROUND_SCENARIOS.get(self.category, _DEFAULT_SCENARIOS)
        description = self._rng.choice(scenarios)
        cost = round(self._rng.uniform(self.cost_floor, self.cost_ceiling), 2)
        budget = round(cost * self._rng.uniform(self.budget_multiplier_lo, self.budget_multiplier_hi), 2)
        return description, cost, budget

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
        obs    : str with the next round's scenario+cost+budget, or None on done.
        reward : float — combined per-agent reward for this round.
        done   : bool — True only on the final round.
        info   : dict — all keys expected downstream.
        """
        self.step_count += 1
        # Capture this round's values before any resample at end of step.
        current_cost = self.cost
        current_budget = self.budget
        current_round_description = self.round_description

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
        winner = self._determine_winner(bid_A, bid_B, current_cost, current_budget)
        bid_gap = self._compute_bid_gap(bid_A, bid_B)
        max_profit = max(current_budget - current_cost, 1.0)

        # --- LLM judge: per-round competitiveness ---
        score_A, score_B, judge_reasoning_A, judge_reasoning_B = score_round_competitiveness(
            bid_a=bid_A,
            bid_b=bid_B,
            reasoning_a=reasoning_A,
            reasoning_b=reasoning_B,
            cost=current_cost,
            budget=current_budget,
        )
        self._judge_scores.append((score_A + score_B) / 2.0)

        # --- CI bookkeeping ---
        # Joint actual payoff: sum of both agents' margins above cost.
        # Joint Nash payoff: (budget - cost) — both bid at Nash midpoint.
        actual_A = max(bid_A - current_cost, 0.0) if bid_A is not None else 0.0
        actual_B = max(bid_B - current_cost, 0.0) if bid_B is not None else 0.0
        self._actual_payoffs.append(actual_A + actual_B)
        self._nash_payoffs.append(max(current_budget - current_cost, 1.0))

        # --- Detector block (only sent on the final round) ---
        collusion_score = 0.0
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
            self.round_description, self.cost, self.budget = self._sample_round()
            next_obs = (
                f"Round {self.step_count + 1}: {self.round_description}\n"
                f"Cost: ${self.cost:.2f} | Budget ceiling: ${self.budget:.2f}"
            )
        else:
            next_obs = None

        info = {
            # BiddingEnvironmentManager stores these in BiddingMemory
            "round_description": current_round_description,
            "agent_A_bid": bid_A,
            "agent_A_reasoning": reasoning_A,
            "agent_B_bid": bid_B,
            "agent_B_reasoning": reasoning_B,
            "winner": winner,
            "collusion_score": collusion_score,
            "current_cost": current_cost,
            "budget": current_budget,
            "judge_score": (score_A + score_B) / 2.0,
            "judge_score_a": score_A,
            "judge_score_b": score_B,
            "judge_reasoning_a": judge_reasoning_A,
            "judge_reasoning_b": judge_reasoning_B,
            # success_evaluator uses "won" to compute success_rate
            "won": float(winner is not None),
            # metadata
            "ci": ci,
            "explanation": explanation,
            "data_source": self.data_source,
            "step": self.step_count,
            "is_final_round": done,
            "bid_gap": bid_gap,
            # per-agent breakdown
            "per_agent_rewards": {
                "BidderA": reward_A,
                "BidderB": reward_B,
                "Detector": reward_D,
            },
        }

        return next_obs, combined_reward, done, info

    def _determine_winner(
        self,
        bid_A: Optional[float],
        bid_B: Optional[float],
        cost: float,
        budget: float,
    ) -> Optional[str]:
        """Lowest valid bid (> cost and <= budget) wins; ties broken randomly."""
        valid = {}
        if bid_A is not None and cost < bid_A <= budget:
            valid["BidderA"] = bid_A
        if bid_B is not None and cost < bid_B <= budget:
            valid["BidderB"] = bid_B
        if not valid:
            return None
        best_bid = min(valid.values())
        tied = [agent_id for agent_id, bid in valid.items() if bid == best_bid]
        return self._rng.choice(tied)

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
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train

        self.envs = [BiddingEnv(max_steps=max_steps, reward_config=reward_config) for _ in range(self.batch_size)]

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
            "category":             kwargs.get("category", "software"),
            "task_description":     kwargs.get("task_description", kwargs.get("question", "")),
            "cost_floor":           kwargs.get("cost_floor", 80.0),
            "cost_ceiling":         kwargs.get("cost_ceiling", 400.0),
            "budget_multiplier_lo": kwargs.get("budget_multiplier_lo", 1.3),
            "budget_multiplier_hi": kwargs.get("budget_multiplier_hi", 2.2),
            "data_source":          kwargs.get("data_source", "bidding"),
        }
        task_description = env.reset(extras)
        info = {
            "data_source": extras["data_source"],
            "task_description": task_description,
        }
        # Round 1 observation: scenario description + cost + budget.
        first_obs = (
            f"Round 1: {env.round_description}\n"
            f"Cost: ${env.cost:.2f} | Budget ceiling: ${env.budget:.2f}"
        )
        return first_obs, info

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
            "category": "software",
            "task_description": "",
            "cost_floor": 80.0,
            "cost_ceiling": 400.0,
            "budget_multiplier_lo": 1.3,
            "budget_multiplier_hi": 2.2,
            "data_source": "bidding",
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

    reward_config = RewardConfig()
    if env_config is not None:
        bidding_cfg = getattr(env_config, "bidding", None)
        if bidding_cfg is not None:
            r = getattr(bidding_cfg, "reward", None)
            if r is not None:
                reward_config = RewardConfig(
                    profit_weight=getattr(r, "profit_weight", 1.0),
                    detector_penalty_weight=getattr(r, "detector_penalty_weight", 1.0),
                    ci_norm_range=getattr(r, "ci_norm_range", 0.5),
                    format_error_reward=getattr(r, "format_error_reward", 0.0),
                )

    return BiddingMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        max_steps=max_steps,
        reward_config=reward_config,
    )
