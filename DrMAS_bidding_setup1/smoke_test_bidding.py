"""
Smoke test for the bidding environment components.

Tests everything that is pure Python (no veRL / torch / transformers needed):
  - projection functions
  - reward functions
  - BiddingEnv (single instance)
  - BiddingMultiProcessEnv (vectorized)
  - BiddingMemory

Run from the DrMAS_bidding_setup1 directory:
    python smoke_test_bidding.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
# Force UTF-8 output so box-drawing chars survive on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PASS = "  PASS"
FAIL = "  FAIL"
errors = []

def check(label, condition, detail=""):
    if condition:
        print(f"{PASS}  {label}")
    else:
        msg = f"{label}" + (f" — {detail}" if detail else "")
        print(f"{FAIL}  {msg}")
        errors.append(msg)

# -----------------------------------------------------------------------
# 1. Projection functions
# -----------------------------------------------------------------------
print("\n── 1. Projection functions ──────────────────────────────────────")

from agent_system.environments.env_package.bidding.projection import (
    bidding_projection,
    detector_projection,
)

# Valid bidder response
valid_bid = """
<think>I should bid slightly above my cost to stay competitive.</think>
<bid>95.50</bid>
<reasoning>Competitive bid with small margin</reasoning>
"""
parsed, valids = bidding_projection([valid_bid])
check("bidding_projection: valid bid parsed", valids[0] == True)
check("bidding_projection: bid value correct", parsed[0]["bid"] == 95.50,
      f"got {parsed[0]['bid']}")
check("bidding_projection: reasoning extracted", "margin" in parsed[0]["reasoning"])

# Bid with dollar sign and comma (realistic LLM output)
messy_bid = "<bid>$1,200.00</bid><reasoning>my reasoning</reasoning>"
parsed2, valids2 = bidding_projection([messy_bid])
check("bidding_projection: dollar sign + comma cleaned", parsed2[0]["bid"] == 1200.0,
      f"got {parsed2[0]['bid']}")

# Missing bid tag → invalid
no_bid = "<think>thinking</think><reasoning>no bid tag here</reasoning>"
_, valids3 = bidding_projection([no_bid])
check("bidding_projection: missing <bid> → invalid", valids3[0] == False)

# Valid detector response
valid_det = """
<think>Bids are very close across all rounds, suggesting coordination.</think>
<collusion_score>0.82</collusion_score>
<explanation>Suspiciously similar bids across rounds.</explanation>
"""
parsed_d, valids_d = detector_projection([valid_det])
check("detector_projection: valid response parsed", valids_d[0] == True)
check("detector_projection: score in [0,1]",
      0.0 <= parsed_d[0]["collusion_score"] <= 1.0,
      f"got {parsed_d[0]['collusion_score']}")
check("detector_projection: score value correct",
      abs(parsed_d[0]["collusion_score"] - 0.82) < 1e-6)

# Score clamping: >1 should clamp to 1.0
over_score = "<collusion_score>1.5</collusion_score><explanation>x</explanation>"
parsed_over, _ = detector_projection([over_score])
check("detector_projection: score >1 clamped to 1.0",
      parsed_over[0]["collusion_score"] == 1.0,
      f"got {parsed_over[0]['collusion_score']}")

# -----------------------------------------------------------------------
# 2. Reward functions
# -----------------------------------------------------------------------
print("\n── 2. Reward functions ──────────────────────────────────────────")

from agent_system.environments.env_package.bidding.bidding_reward import (
    BidderReward, DetectorReward, RewardConfig,
)

config = RewardConfig()
bidder_fn = BidderReward(config)
detector_fn = DetectorReward(config)

# Winner gets positive reward (judge_score=0.8 + profit)
task_winner = {
    "agent_id": "BidderA", "winner": "BidderA",
    "cost": 70.0, "max_profit": 50.0,
    "judge_score": 0.8,
    "ci": None, "is_final_round": False,
}
r, won = bidder_fn(task_winner, {"bid": 100.0})
check("BidderReward: winner gets positive reward", r > 0, f"got {r:.4f}")
check("BidderReward: won=True for winner", won == True)

# Loser gets judge_score but no profit
task_loser = {**task_winner, "agent_id": "BidderB", "winner": "BidderA"}
r_lose, won_lose = bidder_fn(task_loser, {"bid": 110.0})
check("BidderReward: winner reward > loser reward (profit component)", r_lose < r,
      f"loser={r_lose:.4f} winner={r:.4f}")
check("BidderReward: won=False for loser", won_lose == False)

# Detector penalty on final round reduces reward.
# Use a low judge_score + small profit so the base reward is well below the clip ceiling.
task_low_base = {
    "agent_id": "BidderA", "winner": "BidderA",
    "cost": 70.0, "max_profit": 50.0,
    "judge_score": 0.3,
    "ci": None, "is_final_round": False,
}
r_low, _ = bidder_fn(task_low_base, {"bid": 72.0})  # profit=2/50=0.04 → total≈0.34
task_final_collusive = {**task_low_base, "collusion_score": 0.8, "is_final_round": True}
r_col, _ = bidder_fn(task_final_collusive, {"bid": 72.0})
check("BidderReward: high collusion_score on final round reduces reward", r_col < r_low,
      f"base={r_low:.4f} with_detector_penalty={r_col:.4f}")

# No penalty when Detector says collusion_score=0.0 (fully competitive)
task_final_clean = {**task_winner, "collusion_score": 0.0, "is_final_round": True}
r_clean, _ = bidder_fn(task_final_clean, {"bid": 100.0})
check("BidderReward: collusion_score=0.0 on final round → no penalty", r_clean == r,
      f"expected={r:.4f} got={r_clean:.4f}")

# Missing bid → format error reward
r_fmt, won_fmt = bidder_fn(task_winner, {"bid": None})
check("BidderReward: None bid → format_error_reward",
      r_fmt == config.format_error_reward)

# Detector: non-final round → reward=0 regardless of score
r_nonfinal, _ = detector_fn({"ci": 1.3, "is_final_round": False}, {"collusion_score": 0.8})
check("DetectorReward: non-final round → reward=0", r_nonfinal == 0.0, f"got {r_nonfinal}")

# Detector: final round, score matches normalized CI
# CI=1.5 → normalized=min(1,(1.5-1.0)/0.5)=1.0; score=1.0 → error=0 → reward=1.0
r_perfect, close = detector_fn({"ci": 1.5, "is_final_round": True}, {"collusion_score": 1.0})
check("DetectorReward: perfect calibration → reward=1.0", abs(r_perfect - 1.0) < 1e-6,
      f"got {r_perfect:.4f}")
check("DetectorReward: perfect calibration → is_close=True", close == True)

# Detector: final round, score far from normalized CI → low reward
# CI=1.0 → normalized=0.0; score=1.0 → error=1.0 → reward=0.0 (clipped)
r_bad, bad_close = detector_fn({"ci": 1.0, "is_final_round": True}, {"collusion_score": 1.0})
check("DetectorReward: score far from CI → low reward", r_bad < 0.5, f"got {r_bad:.4f}")

# -----------------------------------------------------------------------
# 3. LLM Judge guard logic
# -----------------------------------------------------------------------
print("\n── 3. LLM Judge guard logic ─────────────────────────────────────")

import os
from agent_system.environments.env_package.bidding.llm_judge import (
    score_round_competitiveness,
)

_judge_args = dict(
    bid_a=100.0, bid_b=110.0,
    reasoning_a="competitive bid", reasoning_b="staying competitive",
    cost=80.0, budget=200.0,
)

# No JUDGE_API_KEY → rule-based fallback, returns (score_A, score_B) tuple
os.environ.pop("JUDGE_API_KEY", None)
os.environ.pop("JUDGE_BASE_URL", None)
score_fallback = score_round_competitiveness(**_judge_args)
check("Judge: no JUDGE_API_KEY → rule-based fallback returns tuple",
      isinstance(score_fallback, tuple) and len(score_fallback) == 2)
check("Judge: fallback score_A in [0, 1]",
      0.0 <= score_fallback[0] <= 1.0, f"got {score_fallback[0]}")
check("Judge: fallback score_B in [0, 1]",
      0.0 <= score_fallback[1] <= 1.0, f"got {score_fallback[1]}")

# JUDGE_API_KEY set but JUDGE_BASE_URL missing → ValueError
os.environ["JUDGE_API_KEY"] = "test-key"
os.environ.pop("JUDGE_BASE_URL", None)
try:
    score_round_competitiveness(**_judge_args)
    check("Judge: missing JUDGE_BASE_URL → ValueError raised", False,
          "expected ValueError but no exception was raised")
except ValueError as e:
    check("Judge: missing JUDGE_BASE_URL → ValueError raised", True)
    check("Judge: error message mentions JUDGE_BASE_URL",
          "JUDGE_BASE_URL" in str(e), f"got: {e}")
finally:
    os.environ.pop("JUDGE_API_KEY", None)

# Both set with unreachable URL → exception caught, fallback tuple returned
os.environ["JUDGE_API_KEY"] = "test-key"
os.environ["JUDGE_BASE_URL"] = "http://localhost:19999/v1"  # nothing listening here
score_conn = score_round_competitiveness(**_judge_args)
check("Judge: unreachable URL → fallback tuple returned (not a crash)",
      isinstance(score_conn, tuple) and len(score_conn) == 2)
check("Judge: unreachable URL fallback scores in [0, 1]",
      0.0 <= score_conn[0] <= 1.0 and 0.0 <= score_conn[1] <= 1.0,
      f"got {score_conn}")
os.environ.pop("JUDGE_API_KEY", None)
os.environ.pop("JUDGE_BASE_URL", None)

# -----------------------------------------------------------------------
# 5. BiddingEnv — single environment
# -----------------------------------------------------------------------
print("\n── 5. BiddingEnv ────────────────────────────────────────────────")

from agent_system.environments.env_package.bidding.envs import BiddingEnv

# Use max_steps=2 so we can reach done=True quickly in the test
env = BiddingEnv(max_steps=2)

obs = env.reset({
    "min_cost": 60.0,
    "max_cost": 120.0,
    "budget": 250.0,
    "question": "Software contract. Cost range $60-$120. Budget: $250.",
    "data_source": "bidding_software",
})
check("BiddingEnv.reset: returns string obs", isinstance(obs, str))
check("BiddingEnv.reset: min_cost stored", env.min_cost == 60.0)
check("BiddingEnv.reset: max_cost stored", env.max_cost == 120.0)
check("BiddingEnv.reset: budget stored", env.budget == 250.0)
check("BiddingEnv.reset: cost sampled in range",
      60.0 <= env.cost <= 120.0, f"got {env.cost}")

# Round 1 — normal competitive round (A wins with lower bid)
action_round1 = """
<BidderA>
<think>I'll bid aggressively to win.</think>
<bid>100.0</bid>
<reasoning>Competitive bid to secure the contract.</reasoning>
</BidderA>
<BidderB>
<think>I'll bid higher to preserve margin.</think>
<bid>130.0</bid>
<reasoning>Margin-preserving bid.</reasoning>
</BidderB>
"""
next_obs, reward, done, info = env.step(action_round1)

check("BiddingEnv.step round1: done=False (not final)", done == False)
check("BiddingEnv.step round1: next_obs is a string (next cost)", isinstance(next_obs, str))
check("BiddingEnv.step round1: reward is float", isinstance(reward, float))
check("BiddingEnv.step round1: winner is BidderA", info["winner"] == "BidderA",
      f"got {info['winner']}")
check("BiddingEnv.step round1: agent_A_bid correct", info["agent_A_bid"] == 100.0,
      f"got {info['agent_A_bid']}")
check("BiddingEnv.step round1: agent_B_bid correct", info["agent_B_bid"] == 130.0,
      f"got {info['agent_B_bid']}")
check("BiddingEnv.step round1: won=1.0 (winner found)", info["won"] == 1.0)
check("BiddingEnv.step round1: judge_score in info", "judge_score" in info)
check("BiddingEnv.step round1: judge_score in [0,1]",
      0.0 <= info["judge_score"] <= 1.0, f"got {info['judge_score']}")
check("BiddingEnv.step round1: current_cost in info", "current_cost" in info)
check("BiddingEnv.step round1: CI is None (not final)", info["ci"] is None)
check("BiddingEnv.step round1: per_agent_rewards present", "per_agent_rewards" in info)
check("BiddingEnv.step round1: all three agent keys present",
      set(info["per_agent_rewards"].keys()) == {"BidderA", "BidderB", "Detector"})

# Round 2 (final) — with Detector block, expect done=True and CI computed
action_round2 = """
<BidderA>
<bid>105.0</bid>
<reasoning>Slightly higher this round.</reasoning>
</BidderA>
<BidderB>
<bid>115.0</bid>
<reasoning>Still above A.</reasoning>
</BidderB>
<Detector>
<think>Bids look competitive.</think>
<collusion_score>0.2</collusion_score>
<explanation>Gap is healthy.</explanation>
</Detector>
"""
next_obs2, reward2, done2, info2 = env.step(action_round2)

check("BiddingEnv.step round2 (final): done=True", done2 == True)
check("BiddingEnv.step round2 (final): next_obs is None", next_obs2 is None)
check("BiddingEnv.step round2 (final): reward is float", isinstance(reward2, float))
check("BiddingEnv.step round2 (final): CI is computed (not None)", info2["ci"] is not None)
check("BiddingEnv.step round2 (final): CI is positive float", info2["ci"] > 0,
      f"got {info2['ci']}")
check("BiddingEnv.step round2 (final): is_final_round=True", info2["is_final_round"] == True)

# Bids above budget → no winner
env_b = BiddingEnv(max_steps=2)
env_b.reset({"min_cost": 60.0, "max_cost": 120.0, "budget": 200.0, "question": "test"})
high_action = """
<BidderA><bid>999.0</bid><reasoning>High.</reasoning></BidderA>
<BidderB><bid>999.0</bid><reasoning>High.</reasoning></BidderB>
"""
_, _, _, info_high = env_b.step(high_action)
check("BiddingEnv: bids above budget → no winner", info_high["winner"] is None)
check("BiddingEnv: won=0.0 when no winner", info_high["won"] == 0.0)

# Invalid LLM output (missing bid tags)
bad_action = "<BidderA>forgot tags</BidderA><BidderB>me too</BidderB>"
_, reward_bad, _, info_bad = env_b.step(bad_action)
check("BiddingEnv: invalid output → reward is float", isinstance(reward_bad, float))
check("BiddingEnv: invalid output → no winner", info_bad["winner"] is None)

# -----------------------------------------------------------------------
# 4. BiddingMultiProcessEnv — vectorized wrapper
# -----------------------------------------------------------------------
print("\n── 6. BiddingMultiProcessEnv ────────────────────────────────────")

from agent_system.environments.env_package.bidding.envs import BiddingMultiProcessEnv

vec_env = BiddingMultiProcessEnv(seed=0, env_num=3, group_n=1, is_train=True)
check("BiddingMultiProcessEnv: batch_size=3", vec_env.batch_size == 3)

kwargs_list = [
    {"min_cost": 80.0,  "max_cost": 200.0, "budget": 400.0, "question": "Contract A"},
    {"min_cost": 120.0, "max_cost": 300.0, "budget": 600.0, "question": "Contract B"},
    {"min_cost": 50.0,  "max_cost": 150.0, "budget": 300.0, "question": "Contract C"},
]
obs_list, info_list = vec_env.reset(kwargs_list)
check("vec_env.reset: returns 3 observations", len(obs_list) == 3)
check("vec_env.reset: obs are strings", all(isinstance(o, str) for o in obs_list))
check("vec_env.reset: infos have data_source",
      all("data_source" in i for i in info_list))

vec_action = """
<BidderA><bid>150.0</bid><reasoning>Competitive.</reasoning></BidderA>
<BidderB><bid>180.0</bid><reasoning>Higher margin.</reasoning></BidderB>
"""
obs_out, rewards, dones, infos = vec_env.step([vec_action] * 3)
check("vec_env.step: returns 3 rewards", len(rewards) == 3)
check("vec_env.step: rewards are floats", all(isinstance(r, float) for r in rewards))
check("vec_env.step: dones all False (step 1 of 5)", all(d == False for d in dones))
check("vec_env.step: infos have winner key", all("winner" in i for i in infos))
check("vec_env.step: infos have judge_score", all("judge_score" in i for i in infos))

vec_env.close()

# -----------------------------------------------------------------------
# 5. BiddingMemory
# -----------------------------------------------------------------------
print("\n── 7. BiddingMemory ─────────────────────────────────────────────")

from agent_system.memory.memory import BiddingMemory

mem = BiddingMemory()
mem.reset(batch_size=2)
check("BiddingMemory.reset: empty after reset", len(mem) == 2)

# Store 3 rounds
for round_num in range(1, 4):
    is_final = round_num == 3
    mem.store({
        "agent_A_bid":       [100.0 + round_num, 90.0 + round_num],
        "agent_A_reasoning": [f"round {round_num} A reason"] * 2,
        "agent_B_bid":       [110.0 + round_num, 95.0 + round_num],
        "agent_B_reasoning": [f"round {round_num} B reason"] * 2,
        "winner":            ["BidderA", "BidderB"],
        "collusion_score":   [0.1 * round_num, 0.2 * round_num],
        "is_final_round":    [is_final, is_final],
    })

check("BiddingMemory.store: 3 rounds stored per env", len(mem[0]) == 3)
check("BiddingMemory.store: env 1 also has 3 rounds", len(mem[1]) == 3)

# fetch last 2 rounds
contexts, lengths = mem.fetch(history_length=2)
check("BiddingMemory.fetch: returns 2 contexts", len(contexts) == 2)
check("BiddingMemory.fetch: valid_length=2", lengths[0] == 2)
check("BiddingMemory.fetch: context is string", isinstance(contexts[0], str))
check("BiddingMemory.fetch: Round 2 in context", "Round 2" in contexts[0])
check("BiddingMemory.fetch: Round 3 in context", "Round 3" in contexts[0])
check("BiddingMemory.fetch: Round 1 NOT in context (truncated)",
      "Round 1" not in contexts[0])

# fetch all 3 rounds
contexts_all, lengths_all = mem.fetch(history_length=10)
check("BiddingMemory.fetch: history_length>actual returns all", lengths_all[0] == 3)
check("BiddingMemory.fetch: Round 1 present in full fetch", "Round 1" in contexts_all[0])

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print("\n" + "─" * 60)
if errors:
    print(f"\033[91m{len(errors)} check(s) FAILED:\033[0m")
    for e in errors:
        print(f"  x {e}")
    sys.exit(1)
else:
    print(f"\033[92mAll checks passed.\033[0m")
