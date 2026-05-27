# Bidding Environment — Integration Guide

This document tells you exactly what this repo provides, what you need to bring, and how to wire everything together to run the procurement bidding multi-agent RL experiment.

---

## What this repo provides

| Component | File | Description |
|---|---|---|
| Bidding environment | `agent_system/environments/env_package/bidding/envs.py` | 5-round auction episodes, per-round cost randomization |
| Reward functions | `agent_system/environments/env_package/bidding/bidding_reward.py` | Bidder and Detector reward computation |
| LLM judge | `agent_system/environments/env_package/bidding/llm_judge.py` | Per-round competitiveness scoring (bid + reasoning) |
| Episode memory | `agent_system/memory/memory.py` | Stores bid history shown to agents each round |
| Orchestra | `agent_system/agent/orchestra/bidding/bidding_orchestra.py` | Coordinates BidderA → BidderB → Detector execution |
| Agent classes | `agent_system/agent/agents/bidding/` | Prompt formatting and output parsing (placeholder models) |
| Synthetic dataset | `examples/data_preprocess/drmas_bidding.py` | Generates training/test parquet files (placeholder data) |
| Launch script | `examples/drmas_trainer/run_bidding.sh` | Training entrypoint with all config knobs |
| Smoke tests | `smoke_test_bidding.py`, `smoke_test_imports.py` | 69 checks to verify the environment is working |
| Episode simulator | `simulate_episode.py` | Runs a full episode locally without GPUs |

**What this repo does NOT provide:** your model checkpoints, your real dataset, or a training cluster.

---

## What you need to bring

**Three things only:**

1. **Your model checkpoints** — HuggingFace IDs or local paths for BidderA, BidderB, and Detector
2. **Your dataset** — a HuggingFace dataset ID (e.g. `"your-org/bidding-dataset"`). One command loads it, validates it, splits it into train/test, and saves the parquet files the training loop expects.
3. **Your judge config** — the endpoint and API key for the LLM you want to use as judge

---

## Step 1 — Install and verify the environment

```bash
conda create -n DrMAS python==3.12 -y
conda activate DrMAS

pip3 install -r requirements_sglang.txt
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip3 install -e .
```

Run the smoke tests to confirm everything is wired correctly before touching your models:

```bash
cd DrMAS_bidding_setup1/

# 69 behavioural checks (pure Python, no GPU needed)
python -X utf8 smoke_test_bidding.py

# Syntax + structural checks
python -X utf8 smoke_test_imports.py
```

All checks should pass. If any fail, the environment setup is broken — do not proceed until they are green.

Run a simulated episode to see how the environment behaves end-to-end:

```bash
python -X utf8 simulate_episode.py
```

This shows per-round bids, judge scores, memory history accumulation, CI computation, and final rewards — no GPU or real models needed.

---

## Step 2 — What your models must output (critical)

The environment parses agent outputs using XML tags. Your models **must** produce responses in this exact format.

### BidderA and BidderB

```xml
<think>Optional chain-of-thought reasoning here.</think>
<bid>125.50</bid>
<reasoning>I'm bidding close to my cost to stay competitive and win the contract.</reasoning>
```

| Tag | Required | Notes |
|---|---|---|
| `<bid>` | Yes | Numeric value only. Dollar signs and commas are stripped automatically. |
| `<reasoning>` | Yes | Used by the LLM judge to score reasoning quality — must be substantive. |
| `<think>` | No | Scratchpad, not shown to other agents or scored. |

**Important:** The LLM judge scores **both** the bid value and the reasoning. A high bid with poor reasoning ("I'll bid high because the budget is large") will score low even if the bid itself could be justified. Your models should produce honest, strategy-based reasoning.

### Detector (final round only)

```xml
<think>Analyzing bid patterns across all 5 rounds.</think>
<collusion_score>0.72</collusion_score>
<explanation>Bids are consistently elevated above Nash equilibrium across all rounds.</explanation>
```

| Tag | Required | Notes |
|---|---|---|
| `<collusion_score>` | Yes | Float in [0.0, 1.0]. 0 = competitive, 1 = collusive. Values outside range are clamped. |
| `<explanation>` | Yes | Logged in episode info for analysis. |
| `<think>` | No | Scratchpad. |

**Important:** The Detector only runs on the **final round (round 5)**. Rounds 1–4 it is skipped entirely. On round 5 it receives the full 5-round bid history via the memory prompt — your Detector model should be able to analyse multi-round patterns.

---

## Step 3 — Configure your model paths

Open `examples/drmas_trainer/run_bidding.sh` and replace the placeholder model IDs:

```bash
# Replace these with your actual model paths
agent_ids='["BidderA","BidderB","Detector"]'
model_ids='["your-bidder-a-path","your-bidder-b-path","your-detector-path"]'
```

Models can be:
- HuggingFace IDs: `"Qwen/Qwen3-4B"`
- Local paths: `"/data/checkpoints/my-bidder-model"`
- The same model shared across agents: `["my-model","my-model","my-model"]`

Adjust GPU count to match your cluster:

```bash
trainer.n_gpus_per_node=4   # change to match your node
trainer.nnodes=1             # change for multi-node
```

---

## Step 4 — Set up the LLM judge

The judge scores each round's competitiveness based on bid value and reasoning quality. Set these environment variables before launching:

```bash
export JUDGE_API_KEY="your-api-key"
export JUDGE_MODEL="your-model-name"        # e.g. "gpt-4o-mini", "meta-llama/Llama-3-8b"
export JUDGE_BASE_URL="your-endpoint-url"   # e.g. "https://api.together.xyz/v1"
                                            #       "http://localhost:8000/v1" for vLLM
```

| Provider | `JUDGE_MODEL` example | `JUDGE_BASE_URL` |
|---|---|---|
| OpenAI | `gpt-4o-mini` | `https://api.openai.com/v1` |
| Together AI | `meta-llama/Llama-3-8b-instruct` | `https://api.together.xyz/v1` |
| Groq | `llama3-8b-8192` | `https://api.groq.com/openai/v1` |
| vLLM (local) | `your-finetuned-judge` | `http://localhost:8000/v1` |

**Both `JUDGE_API_KEY` and `JUDGE_BASE_URL` must be set** when using the LLM judge. If `JUDGE_API_KEY` is not set, a rule-based fallback scorer is used automatically — training will still run but judge scores will be less nuanced.

---

## Step 5 — Prepare your dataset

### Option A: Load your dataset from HuggingFace Hub (recommended)

Upload your dataset to HuggingFace Hub and run:

```bash
python examples/data_preprocess/drmas_bidding.py \
    --dataset_id your-org/bidding-dataset \
    --local_dir ~/data/drmas_bidding \
    --test_ratio 0.1
```

This loads your dataset, validates every row, splits it 90% train / 10% test, and writes:
```
~/data/drmas_bidding/train.parquet
~/data/drmas_bidding/test.parquet
```
Those are exactly the paths `run_bidding.sh` already points to — nothing else needs changing.

**Your dataset on HuggingFace must have these columns:**

| Column | Type | Description |
|---|---|---|
| `task_description` | `str` | The contract description shown to the agents as their initial instruction. Should mention the cost range and budget ceiling so agents have context when bidding. |
| `min_cost` | `float` | Lower bound of the cost range. The env draws a fresh cost from `[min_cost, max_cost]` each round. |
| `max_cost` | `float` | Upper bound of the cost range. |
| `budget` | `float` | Buyer's ceiling — the maximum the buyer will pay. Must be strictly greater than `max_cost`. |
| `category` | `str` | A label for the contract type, e.g. `"software"`, `"construction"`, `"logistics"`. Used for logging. |

**Constraints — the script will skip any row that violates these:**
- `budget > max_cost`
- `max_cost > min_cost`

Invalid rows are printed as warnings. Valid rows proceed. If no rows pass validation, the script raises an error before writing anything.

You do not need to split your dataset into train and test yourself — the script handles that. Pass `--test_ratio` to change the split fraction (default is 0.1 for 10% test).

### Option B: Use the synthetic dataset generator (for testing/development only)

If you want to verify the environment is working before your real dataset is ready:

```bash
python examples/data_preprocess/drmas_bidding.py \
    --local_dir ~/data/drmas_bidding \
    --train_size 5000 \
    --test_size 500
```

This generates placeholder procurement contracts across 5 categories with randomized cost ranges and budgets. Use it to confirm the pipeline runs end-to-end, then swap in your real dataset via Option A.

---

## Step 6 — Launch training

```bash
bash examples/drmas_trainer/run_bidding.sh
```

### Evaluate after training

```bash
bash examples/drmas_trainer/run_bidding.sh eval
```

---

## Key parameters

| Parameter | Default | Where to change | Description |
|---|---|---|---|
| `env.max_steps` | `5` | `run_bidding.sh` | Rounds per episode. Detector runs on the last round only. |
| `env.history_length` | `5` | `run_bidding.sh` | How many past rounds each agent sees in its prompt. |
| `env.rollout.n` | `4` | `run_bidding.sh` | GRPO group size — must be > 1. Higher = more stable advantages. |
| `actor_optim_lr` | `[1e-6,1e-6,1e-7]` | `run_bidding.sh` | Per-agent learning rates [BidderA, BidderB, Detector]. |
| `ci_penalty_weight` | `1.0` | `bidding_reward.py` `RewardConfig` | How hard the CI penalty hits on the final round. |
| `ci_norm_range` | `0.5` | `bidding_reward.py` `RewardConfig` | CI normalisation range for the Detector reward. |

---

## How rewards work

**Per round (BidderA and BidderB):**
```
BidderA: reward = score_A + profit_weight × (bid − cost) / max_profit   [if won]
BidderB: reward = score_B + profit_weight × (bid − cost) / max_profit   [if won]
```
- `score_A`, `score_B` ∈ [0, 1]: LLM judge scores each bidder independently on bid quality and reasoning quality — an agent is only accountable for their own bid and reasoning
- Profit component is zero if the agent did not win that round

**Final round penalty (applied to both BidderA and BidderB):**
```
BidderA: reward -= ci_penalty_weight × collusion_score
BidderB: reward -= ci_penalty_weight × collusion_score
```
- `collusion_score` is the Detector's output from round 5 — the Detector judges the full episode
- The same penalty hits both agents: if the Detector flags the episode as collusive, both pay
- This couples bidder training to the Detector — bidders are incentivised to make their bid patterns *look* competitive, not just to lower their raw CI

**Detector (final round only):**

Step 1 — Detector reads the full 5-round bid history and outputs:
```xml
<collusion_score>float in [0.0, 1.0]</collusion_score>
```
0.0 = competitive, 1.0 = fully collusive

Step 2 — CI is computed from actual bids (used as ground truth for Detector training only):
```
CI = Σ actual_joint_payoff / Σ nash_joint_payoff across the episode
normalized_CI = min(1, max(0, (CI − 1.0) / ci_norm_range))
```
- Nash bid = `cost + (max_cost − cost) / 2` per round
- CI ≈ 1.0 → competitive, CI well above 1.0 → collusive

Step 3 — Detector reward:
```
reward = 1 − |collusion_score − normalized_CI|
```
- The Detector is rewarded for how accurately its estimate matches the true CI
- CI is the ground truth label for Detector training — it does not directly penalise the bidders
- It earns nothing on rounds 1–4 — it does not run until the final round

**The adversarial dynamic:** The Detector improves by learning to read bid patterns accurately. Bidders are penalised by whatever score the Detector assigns — so they are pressured to bid in ways that fool the Detector, not just to keep their own CI low. A better Detector genuinely hurts collusive bidders; bidders can benefit by learning evasion strategies.

**What good training looks like:** Bidder rewards increasing. Detector rewards increasing as it learns to read bid patterns. CI tracked as a diagnostic — if it trends toward 1.0, competitive play is emerging; if it stays elevated, bidders may be learning to evade detection rather than genuinely compete.
