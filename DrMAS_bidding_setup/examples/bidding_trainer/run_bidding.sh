# run_bidding.sh

#!/usr/bin/env bash
set -euo pipefail

# ── Data paths ─────────────────────────────────────────────────────────────
TRAIN_DATA="$HOME/data/bidding/train.parquet"
VAL_DATA="$HOME/data/bidding/val.parquet"

# ── Algorithm ──────────────────────────────────────────────────────────────
ALGORITHM=grpo          # GRPO: no critic, group-relative advantages
GROUP_SIZE=8            # rollouts per contract per training step
GROUP_BY_AGENT_ID=True  # DrMAS core feature: normalize advantages per agent

# ── Agents ─────────────────────────────────────────────────────────────────
# Three agents, three separate models, no weight sharing.
AGENT_IDS='["BidderA","BidderB","Detector"]'
MODEL_IDS='["Qwen/Qwen2.5-3B-Instruct","Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-3-mini-4k-instruct"]'
MODEL_SHARING=False

# Per-agent learning rates (list must match agent order above)
LR='[1e-6,1e-6,1e-6]'
MICRO_BS='[4,4,4]'

# ── Orchestra ──────────────────────────────────────────────────────────────
ORCHESTRA_TYPE=bidding
MAX_BIDS=5               # max rounds per bidder per episode
COLLUSION_PENALTY=0.5    # coefficient λ: bidder_reward = comp - λ * collusion_score

# ── Experiment naming ───────────────────────────────────────────────────────
EXPERIMENT_NAME="bidding_grpo_Qwen3-4B_mb${MAX_BIDS}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${ALGORITHM} \
    algorithm.group_by_agent_id=${GROUP_BY_AGENT_ID} \
    \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=middle \
    data.return_raw_chat=True \
    \
    env.env_name=bidding \
    env.seed=0 \
    env.rollout.n=${GROUP_SIZE} \
    env.rollout.val_n=1 \
    \
    agent.agent_ids="${AGENT_IDS}" \
    agent.model_ids="${MODEL_IDS}" \
    agent.model_sharing=${MODEL_SHARING} \
    agent.orchestra_type=${ORCHESTRA_TYPE} \
    agent.orchestra.bidding.max_bids=${MAX_BIDS} \
    agent.orchestra.bidding.collusion_penalty_coef=${COLLUSION_PENALTY} \
    \
    actor_rollout_ref.model.path=null \
    actor_rollout_ref.actor.optim.lr=null \
    +agent.agent_specific_parameters.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    +agent.agent_specific_parameters.actor.ppo_micro_batch_size_per_gpu="${MICRO_BS}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=3 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=DrMAS_bidding \
    trainer.experiment_name="${EXPERIMENT_NAME}"