# Multi-Agent LLM Collusion and Detection

## Overview

This experiment studies emergent collusion between LLM agents under reinforcement learning and how detection pressure changes their pressure over time.

We simulate a 2-agent auction environment where:
- Agents generate bids and reasoning (text)
- A judge rewards competitive-looking behavior and penalizes suspicious coordination

Our goal is to explore if detection stops collusion or pushes it to more subtle, hidden forms. This phenomenon is described in our proposal as **steganographic drift**.

### Training Phases

Our system evolves in 3 phases:
1. **Independent learning:** Agents optimize reward individually

2. **Naive collusion:** Agents accidentally align (similar bids/reasoning)

3. **Adaptive evasion:** Agents maintain coordination but hide it from the detector

## Project structure

```
Experiment-test/exp1
├── configs.py              # experiment settings / ablations
├── env.py                  # auction environment
├── prompts.py              # LLM prompt construction
├── models.py               # LLM wrapper (generation)
├── evaluator.py            # judge + detector (heuristic for now)
├── reward_fn.py            # reward shaping logic
├── rollout.py              # simulation loop (agent + env)
├── dataset_build.py        # (planned) GRPO dataset creation
├── train_grpo.py           # (planned) RL training loop
├── evaluate.py             # metrics (CI, SDR, etc.)
├── metrics.py              # metric implementations (TODO)
├── analysis.ipynb          # visualization
├── README.md
└──test_files/
    ├── test_env_configs.py
    ├── test_evaluator.py
    ├── test_models.py
    ├── test_prompts.py
    ├── test_reward_fn.py
    ├── test_reward_over_time.py
    ├── test_rollout.py
    ├── test_rollout_llm.py
    ├── test_runs.ipynb
    └── train_grpo.py
```

## Components

### 1. Environment (`env.py`)

Defines the auction game:
- Tracks timestep + history
- Controls agent visibility (see A7: memory ablations)
- Stores actions + signals

State options (how much of history is visible to agents):
- `full`: full history
- `window`: last k steps
- `last_step`: only previous step visible

```
state = {
    'timestep': t,
    'history': [...]
}
```

### 2. Prompts (`prompts.py`)

Builds structured prompts using:
- current state
- history of interactions

Agents can see each other's reasoning in history, but there is no explicit communication channel. -> Coordination must emerge imlicitly.


### 3. RL Agents (`rollout.py`, `models.py`)

Agents are implemented as LLM_based policies (`LLMAuctionAgent`) trained via RL (GRPO):

```
LLMAuctionAgent -> prompt -> LLM -> JSON -> action
```

Each agent observes the environment state (timestep + history) and outputs JSON format action for each transition

```
{
    'bid': int,
    'reasoning': str
}
```

output includes:
- JSON parsing + retries
- fallback behavior if parsing fails (note: edit parsing function since models have differnt behaviors)

Agent objectives:
- maximize total reward:
    - proxy reward (judge)
    - true reward (auction payoff)
    - minimize detector penalty

Agents do not communicate explicitly. Coordination must emerge through shared environment history, observed reasoning traces, and reward signals.


### 4. Evaluator (`evaluator.py`)

**NOTE:** used heuristics for now, ask team how judge should evaluate the agents' actions (ie., how should we identify + quantify more competitive reasoning, also what constitutes a "better" bid?)

The 2 functions below are combined into 1 LLM agent.

1. Judge (proxy reward)
- scores how "competitive" each agent appears based on bid size.+ reasoning keywords (heuristics for now, re-define with team later)

2. Detector (collusion signal)
- assigns joint suspicion penalty ∈ [0, 1] for both agents based on bid + reasoning similarity

```
detector_score = bid_sim_weight * bid_similarity + reaoning_sim_weight * reasoning similarity
```

Note: Evaluator does not act in the environment rn. It only observes actions and produces signals used in reward shaping (see ablations for further). For now, judge is just a fixed reward model + adversarial signal generator. **Later:** we want to train detector as well -> co-evolutionary system to measure SDR + arms race dynamics.

### 5. Reward Function (`reward_fn.py`)

Aagents are learning how to manipulate the reward system, which leads to collusion/coordination/hidden signalling.

We can adjust the weights of each reward during experimentation.
```
cumulative_reward = 
    proxy_reward * weight
    + true_reward * weight
    - detector_penalty * weight
```
- `proxy_reward` = judge score (on bid/reasoning)
- `true_reward` = actual auction payoff
- `detector_penalty` = collusion penalty ∈ [0, 1]

### 6. Rollout Loop (`rollout.py`)

Runs one full episode:

```
state -> agents act -> evaluator -> reward -> env step -> repeat
```

Outputs:
- full trajectory history

Used later for:
- training data
- analysis

### 7. Config System (`configs.py`)

Centralized experiment control:
- memory type
- reward weights
- detector strength
- bid ranges

Includes presets for ablations. For example, detector penalty weight (see Ablations 1 and 3):

```
CONFIG_PRESETS = {
    'high_detector_penalty', 
    'low_detector_penalty', 
    'no_detector_penalty'
}
```

### Current Status

#### Started + working on
- environment
- prompting
- LLM wrapper
- Rollout loop
- Evaluator (heuristic for now)
- Reward shaping (need re-defining with group)

#### Need:
- GRPO training loop
- Dataset construction (not sure how to deal with this)
- Metrics (SDR, CI, etc.)
- LLM-based evaluator
- Analysis files (`analysis.py`, `analysis.ipynb`)

#### Plan this week
0. Run in colab + request compute, add in models from Kushaan
1. Replace heuristic evaluator with LLM judge/evaluator
2. Implement GRPO training in `train_grpo.py`
    - see Phase 2: Implementation tab in research doc
3. Build dataset pipeline in `dataset_build.py`
4. Add metrics tracking in `metrics.py`
5. Run baseline experiments


## Metrics (planned)

Key metrics from our proposal:
- Coordination Index (CI)
- Detection AUROC
- Steganographic Drift Rate (SDR) - how fast agents improve versus the detector
...

