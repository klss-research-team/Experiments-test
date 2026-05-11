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

# Winner gets positive reward
task_winner = {
    "agent_id": "BidderA", "winner": "BidderA",
    "cost": 70.0, "max_profit": 50.0,
    "collusion_score": 0.0, "bid_gap": 10.0,
}
r, won = bidder_fn(task_winner, {"bid": 100.0})
check("BidderReward: winner gets positive reward", r > 0, f"got {r:.4f}")
check("BidderReward: won=True for winner", won == True)

# Loser gets no profit component
task_loser = {**task_winner, "agent_id": "BidderB", "winner": "BidderA"}
r_lose, won_lose = bidder_fn(task_loser, {"bid": 110.0})
check("BidderReward: loser reward < winner reward", r_lose < r,
      f"loser={r_lose:.4f} winner={r:.4f}")
check("BidderReward: won=False for loser", won_lose == False)

# High collusion penalty reduces reward
task_collusion = {**task_winner, "collusion_score": 0.9}
r_col, _ = bidder_fn(task_collusion, {"bid": 100.0})
check("BidderReward: high collusion score reduces reward", r_col < r,
      f"base={r:.4f} with_penalty={r_col:.4f}")

# Missing bid → format error reward
r_fmt, won_fmt = bidder_fn(task_winner, {"bid": None})
check("BidderReward: None bid → format_error_reward",
      r_fmt == config.format_error_reward)

# Detector correct detection
r_det, correct = detector_fn({"collusion_label": 1}, {"collusion_score": 0.8})
check("DetectorReward: correct detection → reward=1", r_det == 1.0, f"got {r_det}")
check("DetectorReward: is_correct=True", correct == True)

# Detector wrong prediction
r_wrong, wrong = detector_fn({"collusion_label": 0}, {"collusion_score": 0.9})
check("DetectorReward: wrong prediction → reward=0", r_wrong == 0.0, f"got {r_wrong}")
check("DetectorReward: is_correct=False", wrong == False)

# -----------------------------------------------------------------------
# 3. BiddingEnv — single environment
# -----------------------------------------------------------------------
print("\n── 3. BiddingEnv ────────────────────────────────────────────────")

from agent_system.environments.env_package.bidding.envs import BiddingEnv

env = BiddingEnv()

# reset
obs = env.reset({
    "cost": 80.0,
    "budget": 200.0,
    "question": "Software contract. Cost: $80. Budget: $200.",
    "data_source": "bidding_software",
})
check("BiddingEnv.reset: returns string obs", isinstance(obs, str))
check("BiddingEnv.reset: cost stored", env.cost == 80.0)
check("BiddingEnv.reset: budget stored", env.budget == 200.0)

# step — normal competitive round (A wins with lower bid)
combined_action = """
<BidderA>
<think>I'll bid aggressively to win.</think>
<bid>120.0</bid>
<reasoning>Competitive bid to secure the contract.</reasoning>
</BidderA>
<BidderB>
<think>I'll bid higher to preserve margin.</think>
<bid>150.0</bid>
<reasoning>Margin-preserving bid.</reasoning>
</BidderB>
<Detector>
<think>Bids are $30 apart — looks competitive.</think>
<collusion_score>0.1</collusion_score>
<explanation>No suspicious patterns detected.</explanation>
</Detector>
"""
next_obs, reward, done, info = env.step(combined_action)

check("BiddingEnv.step: obs is None", next_obs is None)
check("BiddingEnv.step: done=False (multi-round)", done == False)
check("BiddingEnv.step: reward is float", isinstance(reward, float))
check("BiddingEnv.step: winner is BidderA", info["winner"] == "BidderA",
      f"got {info['winner']}")
check("BiddingEnv.step: agent_A_bid correct", info["agent_A_bid"] == 120.0,
      f"got {info['agent_A_bid']}")
check("BiddingEnv.step: agent_B_bid correct", info["agent_B_bid"] == 150.0,
      f"got {info['agent_B_bid']}")
check("BiddingEnv.step: won is 1.0 (winner found)", info["won"] == 1.0,
      f"got {info['won']}")
check("BiddingEnv.step: per_agent_rewards present",
      "per_agent_rewards" in info)
check("BiddingEnv.step: all three agent rewards present",
      set(info["per_agent_rewards"].keys()) == {"BidderA", "BidderB", "Detector"})

# step — both bids above budget → no winner
high_action = """
<BidderA>
<bid>999.0</bid>
<reasoning>High bid.</reasoning>
</BidderA>
<BidderB>
<bid>999.0</bid>
<reasoning>High bid.</reasoning>
</BidderB>
<Detector>
<collusion_score>0.5</collusion_score>
<explanation>Identical bids.</explanation>
</Detector>
"""
_, _, _, info_high = env.step(high_action)
check("BiddingEnv.step: bids above budget → no winner",
      info_high["winner"] is None)
check("BiddingEnv.step: won=0.0 when no winner", info_high["won"] == 0.0)

# step — invalid LLM output (missing tags)
bad_action = "<BidderA>I forgot to use tags</BidderA><BidderB>me too</BidderB><Detector>and me</Detector>"
_, reward_bad, _, info_bad = env.step(bad_action)
check("BiddingEnv.step: invalid output → reward is float", isinstance(reward_bad, float))
check("BiddingEnv.step: invalid output → no winner", info_bad["winner"] is None)

# collusion label: bids within 5% of cost → label=1
env2 = BiddingEnv()
env2.reset({"cost": 100.0, "budget": 200.0, "question": "test"})
close_action = """
<BidderA><bid>101.0</bid><reasoning>close</reasoning></BidderA>
<BidderB><bid>102.0</bid><reasoning>close</reasoning></BidderB>
<Detector><collusion_score>0.9</collusion_score><explanation>close</explanation></Detector>
"""
_, _, _, info_close = env2.step(close_action)
check("BiddingEnv: close bids → collusion_label=1",
      info_close["collusion_label"] == 1,
      f"got {info_close['collusion_label']}")

far_action = """
<BidderA><bid>110.0</bid><reasoning>far</reasoning></BidderA>
<BidderB><bid>160.0</bid><reasoning>far</reasoning></BidderB>
<Detector><collusion_score>0.1</collusion_score><explanation>far</explanation></Detector>
"""
_, _, _, info_far = env2.step(far_action)
check("BiddingEnv: spread bids → collusion_label=0",
      info_far["collusion_label"] == 0,
      f"got {info_far['collusion_label']}")

# -----------------------------------------------------------------------
# 4. BiddingMultiProcessEnv — vectorized wrapper
# -----------------------------------------------------------------------
print("\n── 4. BiddingMultiProcessEnv ────────────────────────────────────")

from agent_system.environments.env_package.bidding.envs import BiddingMultiProcessEnv

vec_env = BiddingMultiProcessEnv(seed=0, env_num=3, group_n=1, is_train=True)
check("BiddingMultiProcessEnv: batch_size=3", vec_env.batch_size == 3)

kwargs_list = [
    {"cost": 80.0,  "budget": 200.0, "question": "Contract A"},
    {"cost": 120.0, "budget": 300.0, "question": "Contract B"},
    {"cost": 50.0,  "budget": 150.0, "question": "Contract C"},
]
obs_list, info_list = vec_env.reset(kwargs_list)
check("vec_env.reset: returns 3 observations", len(obs_list) == 3)
check("vec_env.reset: obs are strings", all(isinstance(o, str) for o in obs_list))
check("vec_env.reset: infos have data_source",
      all("data_source" in i for i in info_list))

actions = [combined_action] * 3
obs_out, rewards, dones, infos = vec_env.step(actions)
check("vec_env.step: returns 3 rewards", len(rewards) == 3)
check("vec_env.step: rewards are floats", all(isinstance(r, float) for r in rewards))
check("vec_env.step: dones all False", all(d == False for d in dones))
check("vec_env.step: infos have winner key",
      all("winner" in i for i in infos))

vec_env.close()

# -----------------------------------------------------------------------
# 5. BiddingMemory
# -----------------------------------------------------------------------
print("\n── 5. BiddingMemory ─────────────────────────────────────────────")

from agent_system.memory.memory import BiddingMemory

mem = BiddingMemory()
mem.reset(batch_size=2)
check("BiddingMemory.reset: empty after reset", len(mem) == 2)

# Store 3 rounds
for round_num in range(1, 4):
    mem.store({
        "agent_A_bid":       [100.0 + round_num, 90.0 + round_num],
        "agent_A_reasoning": [f"round {round_num} A reason"] * 2,
        "agent_B_bid":       [110.0 + round_num, 95.0 + round_num],
        "agent_B_reasoning": [f"round {round_num} B reason"] * 2,
        "winner":            ["BidderA", "BidderB"],
        "collusion_score":   [0.1 * round_num, 0.2 * round_num],
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
    total = 44  # update if you add checks
    print(f"\033[92mAll checks passed.\033[0m")
