set -x

# ---------------------------------------------------------------------------
# Usage:
#   bash run_bidding.sh          # training mode (default, 4 GPUs)
#   bash run_bidding.sh quick    # diagnostic run (1 epoch, 4 GPUs)
#   bash run_bidding.sh colab    # single-GPU mode for Google Colab (1 epoch)
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
    total_epochs=5
elif [ "$MODE" == "quick" ]; then
    echo "Running in quick mode (diagnostic — 1 epoch, small batch)"
    VAL_ONLY=False
    TRAIN_DATA="$HOME/data/drmas_bidding/train.parquet"
    VAL_DATA="$HOME/data/drmas_bidding/test.parquet"
    train_data_size=32
    val_data_size=16
    val_group_size=1
    total_epochs=1
elif [ "$MODE" == "colab" ]; then
    echo "Running in Colab mode (1 GPU, 1 epoch)"
    VAL_ONLY=False
    TRAIN_DATA="$HOME/data/drmas_bidding/train.parquet"
    VAL_DATA="$HOME/data/drmas_bidding/test.parquet"
    train_data_size=32
    val_data_size=32
    val_group_size=1
    total_epochs=1
else
    echo "Running in training mode"
    VAL_ONLY=False
    TRAIN_DATA="$HOME/data/drmas_bidding/train.parquet"
    VAL_DATA="$HOME/data/drmas_bidding/test.parquet"
    train_data_size=32
    val_data_size=32
    val_group_size=1
    total_epochs=5
fi

###################### LLM Judge Configuration #############################
# The per-round competitiveness judge calls any OpenAI-compatible LLM API.
# Set these env vars before launching (or export them in your shell profile):
#
#   JUDGE_API_KEY   — your API key (required to enable the LLM judge)
#   JUDGE_MODEL     — model to use, e.g. "gpt-4o-mini", "meta-llama/Llama-3-8b-instruct"
#   JUDGE_BASE_URL  — base URL of the endpoint (leave unset for OpenAI)
#
# Examples:
#   OpenAI  : export JUDGE_API_KEY="sk-..."  JUDGE_MODEL="gpt-4o-mini"
#   Together: export JUDGE_API_KEY="..."  JUDGE_MODEL="meta-llama/..."  JUDGE_BASE_URL="https://api.together.xyz/v1"
#   vLLM    : export JUDGE_API_KEY="token"  JUDGE_MODEL="your-model"  JUDGE_BASE_URL="http://localhost:8000/v1"
#
# JUDGE_API_KEY and JUDGE_MODEL are required — training will raise an error if not set.
# JUDGE_BASE_URL is required for non-OpenAI endpoints (e.g. OpenRouter, vLLM).
#
# OpenRouter example:
#   export JUDGE_API_KEY="sk-or-v1-..."
#   export JUDGE_MODEL="openai/gpt-4o-mini"        # or any OpenRouter model ID
#   export JUDGE_BASE_URL="https://openrouter.ai/api/v1"
: "${JUDGE_API_KEY:=}"
: "${JUDGE_MODEL:=}"
: "${JUDGE_BASE_URL:=}"

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

###################### Reward Weights ######################################
profit_weight=1.0           # scale factor for normalized profit component
detector_penalty_weight=1.0 # scale factor for Detector collusion penalty on final round
ci_norm_range=0.3           # CI normalisation range (CI=1+ci_norm_range → normalized_CI=1.0)

###################### Data / Model Lengths ################################
max_prompt_length=4096
max_response_length=512

###################### Hardware ############################################
n_gpus_per_node=4   # override to 1 automatically in colab mode below
param_offload=False          # offload model weights to CPU RAM (slower, saves GPU VRAM)
gpu_memory_utilization=0.5   # fraction of GPU VRAM reserved for sglang KV cache
max_model_len=null            # null = use model default (40960); override in colab to cap KV cache
free_cache_engine=False       # if True, sglang frees KV cache after rollout (saves memory during training)
enforce_eager=False           # if True, disables CUDA graphs (required when free_cache_engine=True)
use_remove_padding=True       # set to False when flash_attn is unavailable
attn_implementation=flash_attention_2  # override to sdpa when flash_attn is unavailable

###################### Checkpointing #######################################
save_freq=100                          # save every N steps (AWS default)
checkpoint_dir=checkpoints/DrMAS_bidding  # local path for AWS

###################### Training Penalties ##################################
invalid_action_penalty_coef=0.1

###################### LoRA ################################################
lora_rank=0                    # 0 = full fine-tuning; >0 enables LoRA
lora_alpha=0                   # ignored when lora_rank=0; convention: 2×lora_rank
lora_target_modules='["q_proj","k_proj","v_proj","o_proj"]'

###################### Colab Single-GPU Overrides ##########################
# Qwen3-4B with LoRA rank=16 on A100 80GB (167GB system RAM).
# gpu_memory_utilization=0.20 → 3 sglang instances × 16GB = 48GB (model+KV),
# leaving ~32GB free during training. LoRA (rank=16) trains ~0.6% of parameters.
if [ "$MODE" == "colab" ]; then
    n_gpus_per_node=1
    actor_ppo_micro_batch_size_per_gpu='[1,1,1]'
    model_ids='["Qwen/Qwen3-4B","Qwen/Qwen3-4B","Qwen/Qwen3-4B"]'
    param_offload=True
    gpu_memory_utilization=0.20  # 3×16GB=48GB for sglang (model+KV); 32GB free for training
    free_cache_engine=True        # release KV cache after rollout → more headroom during training
    enforce_eager=True            # required when free_cache_engine=True
    max_model_len=5120
    save_freq=5
    checkpoint_dir=/content/checkpoints
    use_remove_padding=False      # flash_attn unavailable on Colab CUDA 12.4
    attn_implementation=sdpa      # use PyTorch native SDPA instead of flash_attention_2
    lora_rank=16
    lora_alpha=32
fi

###################### Experiment Naming ##################################
model_name_tag=$(jq -r '.[]' <<< "$model_ids" | awk -F/ '{print $NF}' | tr '[:upper:]' '[:lower:]' | tr '-' '_' | paste -sd_)
experiment_name="drmas_bidding_share${model_sharing}_${model_name_tag}"

###################### Startup Summary ####################################
echo ""
echo "=== DrMAS Bidding Run ==="
echo "  mode             : $MODE"
echo "  model            : $(jq -r '.[0]' <<< "$model_ids") (all agents)"
echo "  agents           : $agent_ids"
echo "  epochs           : $total_epochs"
echo "  train_data_size  : $train_data_size"
echo "  group_size       : $group_size"
echo "  profit_weight    : $profit_weight"
echo "  detector_penalty : $detector_penalty_weight"
echo "  ci_norm_range    : $ci_norm_range"
echo "  max_response_len : $max_response_length"
echo "  n_gpus_per_node  : $n_gpus_per_node"
echo "  param_offload    : $param_offload"
echo "  gpu_mem_util     : $gpu_memory_utilization"
echo "  max_model_len    : $max_model_len"
echo "  save_freq        : $save_freq"
echo "  checkpoint_dir   : $checkpoint_dir"
echo "  lora_rank        : $lora_rank"
echo "  judge model      : ${JUDGE_MODEL:-NOT SET}"
echo "  judge base url   : ${JUDGE_BASE_URL:-NOT SET}"
echo "========================="
echo ""

###################### Launch #############################################
# Build optional LoRA args (only injected when lora_rank > 0)
lora_args=""
if [ "$lora_rank" -gt 0 ]; then
    lora_args="actor_rollout_ref.model.lora_rank=$lora_rank actor_rollout_ref.model.lora_alpha=$lora_alpha actor_rollout_ref.model.target_modules=$lora_target_modules"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    algorithm.group_by_agent_id=$group_by_agent_id \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='middle' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=null \
    actor_rollout_ref.actor.optim.lr=null \
    +agent.agent_specific_parameters.actor.optim.lr=$actor_optim_lr \
    actor_rollout_ref.model.use_remove_padding=$use_remove_padding \
    +actor_rollout_ref.model.attn_implementation=$attn_implementation \
    $lora_args \
    actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size=True \
    actor_rollout_ref.actor.ppo_mini_update_num=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    +agent.agent_specific_parameters.actor.ppo_micro_batch_size_per_gpu=$actor_ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$param_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.max_model_len=$max_model_len \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=$enforce_eager \
    actor_rollout_ref.rollout.free_cache_engine=$free_cache_engine \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=$invalid_action_penalty_coef \
    env.env_name=bidding \
    env.seed=0 \
    env.max_steps=$env_max_steps \
    env.history_length=$env_history_length \
    env.bidding.reward.profit_weight=$profit_weight \
    env.bidding.reward.detector_penalty_weight=$detector_penalty_weight \
    env.bidding.reward.ci_norm_range=$ci_norm_range \
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
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.default_local_dir=$checkpoint_dir/$experiment_name \
    trainer.save_freq=$save_freq \
    trainer.test_freq=10 \
    trainer.total_epochs=$total_epochs \
    trainer.val_before_train=True \
    trainer.val_only=$VAL_ONLY
