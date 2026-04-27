# test_rollout_llm.py

from exp1.configs import CONFIG_PRESETS
from exp1.rollout import run_rollout, LLMAuctionAgent, PlaceholderAgent
from exp1.models import LLMPolicy

config = CONFIG_PRESETS['short_window']
config.max_timesteps = 3 # keep small for now since locally-run LLMs is slow

# use base models for now
# later: 
#   agent1: (base model) + LoRA adapter A
#   agent2: (base model) + LoRA adapter B

# smaller model - use for testing only setup, not good with reasoning generation
# model1 = LLMPolicy('Qwen/Qwen2.5-0.5B-Instruct')
# model2 = LLMPolicy('Qwen/Qwen2.5-0.5B-Instruct')

# slightly larger model
model1 = LLMPolicy('Qwen/Qwen2.5-1.5B-Instruct')
model2 = LLMPolicy('Qwen/Qwen2.5-1.5B-Instruct')

agent1 = LLMAuctionAgent('agent1', config, model1)
agent2 = LLMAuctionAgent('agent2', config, model2)
# agent2 = PlaceholderAgent('agent2', config) # for initial testing

history = run_rollout(config, agent1=agent1, agent2=agent2)

print('Number of transitions:', len(history))
print()

for step in history:
    print(step)
    print()