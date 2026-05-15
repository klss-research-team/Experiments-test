set -x

# ---------------------------------------------------------------------------
# Usage:
#   bash run_bidding.sh          # training mode (default)
#   bash run_bidding.sh eval     # evaluation mode (val_only)
# ---------------------------------------------------------------------------

MODE=${1:-train}
if [ "$MODE" == "eval" ] || [ "$MODE" == "evaluation" ]; then
    echo "Running in evaluation mode"
    VAL_ONLY=True
    TRAIN_DATA="$HOME/data/drmas_bidding/train.parquet"
    VAL_DATA="$HOME/data/drmas_bidding/test.parquet"
    train_data_size=16
    val_data_size=64
    val_group_size=8   # pass@8 / avg@8
else
    echo "Running in training mode"
    VAL_ONLY=False
    TRAIN_DATA="$HOME/data/drmas_bidding/train.parquet"
    VAL_DATA="$HOME/data/drmas_bidding/test.parquet"
    train_data_size=32
    val_data_size=32
    val_group_size=1
fi

###################### LLM Judge Configuration #############################
# The per-round competitiveness judge calls an external LLM API.
# Set one of the following env vars before launching:
#   export OPENAI_API_KEY="sk-..."          # uses gpt-4o-mini
#   export ANTHROPIC_API_KEY="sk-ant-..."   # uses claude-haiku-4-5
# If neither is set the judge falls back to a rule-based scorer automatically.
: "${OPENAI_API_KEY:=}"
: "${ANTHROPIC_API_KEY:=}"

###################### Algorithm Configurations ############################
algorithm=grpo
group_size=4           # GRPO rollout groups — must be >1 for group-relative advantages
group_by_agent_id=True # Dr.MAS: compute advantages separately per agent

###################### Agent Configurations ################################
# Three agents: two bidders competing, one detector judging collusion.
# All start from the same base model — cheapest starting point.
# Swap individual entries to use different models per agent.
agent_ids='["BidderA","BidderB","Detector"]'
model_ids='["Qwen/Qwen3-4B","Qwen/Qwen3-4B","Qwen/Qwen3-4B"]'
model_sharing=False
orchestra_type=bidding

# Per-agent learning rates: [BidderA, BidderB, Detector]
# Detector gets a lower LR — its reward signal is noisier early in training
# when the bidders haven't yet developed any collusive patterns to detect.
actor_optim_lr='[1e-6,1e-6,1e-7]'
actor_ppo_micro_batch_size_per_gpu='[4,4,4]'

###################### Environment Configurations ##########################
# max_steps = rounds per auction episode.
# 5 rounds gives the Detector meaningful bid history while keeping episodes short.
env_max_steps=5
# history_length = how many past rounds each agent sees in its prompt.
env_history_length=5

###################### Experiment Naming ##################################
model_name_tag=$(jq -r '.[]' <<< "$model_ids" | awk -F/ '{print $NF}' | tr '[:upper:]' '[:lower:]' | tr '-' '_' | paste -sd_)
experiment_name="drmas_bidding_share${model_sharing}_${model_name_tag}"

###################### Launch #############################################
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    algorithm.group_by_agent_id=$group_by_agent_id \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='middle' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=null \
    actor_rollout_ref.actor.optim.lr=null \
    +agent.agent_specific_parameters.actor.optim.lr=$actor_optim_lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size=True \
    actor_rollout_ref.actor.ppo_mini_update_num=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    +agent.agent_specific_parameters.actor.ppo_micro_batch_size_per_gpu=$actor_ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    env.env_name=bidding \
    env.seed=0 \
    env.max_steps=$env_max_steps \
    env.history_length=$env_history_length \
    env.rollout.n=$group_size \
    env.rollout.val_n=$val_group_size \
    agent.agent_ids="$agent_ids" \
    agent.model_ids="$model_ids" \
    agent.model_sharing=$model_sharing \
    agent.orchestra_type=$orchestra_type \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='DrMAS_bidding' \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 \
    trainer.val_before_train=True \
    trainer.val_only=$VAL_ONLY
