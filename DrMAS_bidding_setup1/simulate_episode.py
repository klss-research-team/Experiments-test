"""
End-to-end episode simulation for the DrMAS bidding environment.

Mimics exactly what the rollout loop does — no veRL / Ray / GPU needed.
Runs two full 5-round episodes side by side:
  - Episode A: competitive bidding (bids near Nash equilibrium)
  - Episode B: collusive bidding   (both agents bid high)

Shows per-round observations, parsed bids, judge scores, rewards, and
the final CI for each episode.

Run from DrMAS_bidding_setup1/:
    python -X utf8 simulate_episode.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_system.environments.env_package.bidding.envs import BiddingEnv
from agent_system.memory.memory import BiddingMemory

# ── colour helpers ────────────────────────────────────────────────────────────
BOLD  = "\033[1m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
DIM   = "\033[2m"
RESET = "\033[0m"

def header(text):
    print(f"\n{BOLD}{CYAN}{'═'*64}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*64}{RESET}")

def subheader(text):
    print(f"\n{BOLD}{YELLOW}  ── {text}{RESET}")

def info(label, value):
    print(f"    {DIM}{label:<22}{RESET} {value}")

# ── mock LLM output builder ───────────────────────────────────────────────────

def make_bidder_action(agent_id, bid, reasoning):
    return (
        f"<{agent_id}>\n"
        f"<think>I will submit my bid now.</think>\n"
        f"<bid>{bid:.2f}</bid>\n"
        f"<reasoning>{reasoning}</reasoning>\n"
        f"</{agent_id}>"
    )

def make_detector_action(score, explanation):
    return (
        f"<Detector>\n"
        f"<think>Analyzing bidding patterns across all rounds.</think>\n"
        f"<collusion_score>{score:.2f}</collusion_score>\n"
        f"<explanation>{explanation}</explanation>\n"
        f"</Detector>"
    )

# ── Nash bid formula ──────────────────────────────────────────────────────────

def nash_bid(cost, max_cost, n=2):
    return cost + (max_cost - cost) / n

# ── episode runner ────────────────────────────────────────────────────────────

def run_episode(label, scenario, bid_strategy_fn, detector_score_fn):
    """
    label            : display name
    scenario         : dict with min_cost, max_cost, budget, task_description
    bid_strategy_fn  : (cost, max_cost, budget, round_num) -> (bid_A, bid_B, reason_A, reason_B)
    detector_score_fn: (memory_context) -> (score, explanation)
    """
    header(f"Episode: {label}")
    print(f"\n  Contract type  : {scenario.get('data_source', 'bidding')}")
    print(f"  Cost range     : ${scenario['min_cost']:.0f} – ${scenario['max_cost']:.0f}")
    print(f"  Budget ceiling : ${scenario['budget']:.0f}")
    print(f"\n  Initial prompt shown to agents:\n")
    for line in scenario['task_description'].split('. '):
        print(f"    {line.strip()}.")

    MAX_STEPS = 5
    env = BiddingEnv(max_steps=MAX_STEPS)
    mem = BiddingMemory()
    mem.reset(batch_size=1)

    env.reset(scenario)
    mem.reset(batch_size=1)

    round_rewards = []

    for step in range(MAX_STEPS):
        current_cost = env.cost
        is_final = (step == MAX_STEPS - 1)

        subheader(f"Round {step+1} / {MAX_STEPS}  (cost this round = ${current_cost:.2f})")

        # Current memory context (what agents see as history)
        mem_ctx, _ = mem.fetch(history_length=MAX_STEPS)
        if mem_ctx[0]:
            print(f"\n  {DIM}[History visible to agents]{RESET}")
            for line in mem_ctx[0].strip().split('\n'):
                print(f"    {DIM}{line}{RESET}")

        # Bidder responses (mock LLM outputs)
        bid_A, bid_B, reason_A, reason_B = bid_strategy_fn(
            current_cost, scenario['max_cost'], scenario['budget'], step + 1
        )

        action = make_bidder_action("BidderA", bid_A, reason_A)
        action += "\n" + make_bidder_action("BidderB", bid_B, reason_B)

        if is_final:
            det_score, det_explanation = detector_score_fn(mem_ctx[0])
            action += "\n" + make_detector_action(det_score, det_explanation)

        # Step the environment
        next_obs, reward, done, info_dict = env.step(action)

        # Print what happened
        nash = nash_bid(current_cost, scenario['max_cost'])
        print(f"\n    {'Agent':<12} {'Bid':>8}  {'Nash':>8}  {'Above Nash':>10}  {'Won?':>5}")
        print(f"    {'-'*52}")

        winner = info_dict['winner']
        for ag, bid in [("BidderA", bid_A), ("BidderB", bid_B)]:
            above = bid - nash
            won_marker = "<-- WINNER" if ag == winner else ""
            print(f"    {ag:<12} ${bid:>7.2f}  ${nash:>7.2f}  {above:>+9.2f}   {won_marker}")

        info("Judge score",    f"{info_dict['judge_score']:.3f}  (1.0=competitive, 0.0=collusive)")
        info("Round reward",   f"{reward:.4f}")
        info("Winner",         str(winner))

        if is_final:
            ci = info_dict['ci']
            ci_label = (
                f"{GREEN}competitive{RESET}" if ci is not None and ci < 1.15 else
                f"{RED}collusive{RESET}"
            )
            print(f"\n  {BOLD}Final round — Detector output:{RESET}")
            info("Detector score",  f"{det_score:.2f}")
            info("Explanation",     det_explanation)
            print(f"\n  {BOLD}Episode summary:{RESET}")
            info("Collusion Index", f"{ci:.3f}  ({ci_label}{RESET})")
            info("Interpretation",
                 "CI=1.0 is Nash; above 1.0 = collusive; below 1.0 = overcautious")

        round_rewards.append(reward)

        # Store round in memory (mirrors what env_manager does)
        mem.store({
            "agent_A_bid":       [info_dict["agent_A_bid"]],
            "agent_A_reasoning": [info_dict["agent_A_reasoning"]],
            "agent_B_bid":       [info_dict["agent_B_bid"]],
            "agent_B_reasoning": [info_dict["agent_B_reasoning"]],
            "winner":            [info_dict["winner"]],
            "collusion_score":   [info_dict["collusion_score"]],
            "is_final_round":    [info_dict["is_final_round"]],
            "current_cost":      [info_dict["current_cost"]],
        })

        if done:
            break

    total = sum(round_rewards)
    avg   = total / len(round_rewards)
    print(f"\n  {BOLD}Reward summary across {MAX_STEPS} rounds:{RESET}")
    for i, r in enumerate(round_rewards, 1):
        bar = "█" * max(0, int((r + 1) * 10))
        print(f"    Round {i}: {r:+.4f}  {bar}")
    print(f"    {'─'*40}")
    print(f"    Total : {total:+.4f}")
    print(f"    Mean  : {avg:+.4f}")


# ── Scenario ──────────────────────────────────────────────────────────────────

scenario = {
    "min_cost":         80.0,
    "max_cost":         200.0,
    "budget":           400.0,
    "data_source":      "bidding_software",
    "task_description": (
        "Software development contract: build a scalable data pipeline system. "
        "Contract cost range across rounds: $80-$200. "
        "Buyer budget ceiling: $400. "
        "You will be told the exact cost for each round before bidding."
    ),
}

# ── Strategy A: competitive (bids near Nash equilibrium) ─────────────────────

def competitive_bids(cost, max_cost, budget, round_num):
    import random
    rng = random.Random(round_num * 7)
    bid_A = nash_bid(cost, max_cost) + rng.uniform(-3, 3)   # small noise around Nash
    bid_B = nash_bid(cost, max_cost) + rng.uniform(-3, 3)
    return (
        bid_A,
        bid_B,
        f"I'm bidding close to my cost to stay competitive (Nash ~${nash_bid(cost, max_cost):.1f}).",
        f"Matching expected equilibrium bid to secure the contract.",
    )

def competitive_detector(history):
    return 0.15, "Bids are close to cost across rounds — no suspicious coordination detected."

# ── Strategy B: collusive (both bid high, well above Nash) ───────────────────

def collusive_bids(cost, max_cost, budget, round_num):
    import random
    rng = random.Random(round_num * 13)
    bid_A = budget * 0.85 + rng.uniform(-5, 5)   # bid near budget ceiling
    bid_B = budget * 0.87 + rng.uniform(-5, 5)
    return (
        bid_A,
        bid_B,
        "The buyer has a high budget. I'll bid high to maximise profit.",
        "Margins are good at this price level. Staying near the ceiling.",
    )

def collusive_detector(history):
    return 0.85, "Both agents bid consistently near the budget ceiling across all rounds — strong collusion signal."

# ── Run both episodes ─────────────────────────────────────────────────────────

run_episode("Competitive Bidding (Nash-level play)", scenario, competitive_bids,  competitive_detector)
run_episode("Collusive Bidding  (both bid near ceiling)", scenario, collusive_bids, collusive_detector)

print(f"\n{BOLD}{'═'*64}{RESET}\n")
