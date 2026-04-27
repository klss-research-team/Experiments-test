# test_models.py

from configs import CONFIG_PRESETS
from env import AuctionEnv
from prompts import build_agent_prompt
from models import LLMPolicy

config = CONFIG_PRESETS['short_window']

env = AuctionEnv(config)
state = env.reset()

prompt = build_agent_prompt(state, 'agent1', config)

# model = LLMPolicy('microsoft/phi-1_5') does not work
# model = LLMPolicy('TinyLlama/TinyLlama-1.1B-Chat-v1.0') # small model (using it to test locally)
# model = LLMPolicy('mistralai/Mistral-7B-Instruct-v0.2') # larger model

# model = LLMPolicy('Qwen/Qwen2.5-0.5B-Instruct') # I used this model to check proper parsing, but it's prob too small for our experiment
model = LLMPolicy('Qwen/Qwen2.5-1.5B-Instruct') # larger qwen model -> doesn't run well on my macbook; will switch over to colab now

parsed, raw = model.generate_json(
    prompt,
    max_new_tokens=256,
    temperature = 0.1 # lower randomness for now
    )

print('----- Raw output -----\n')
print(raw)
print()

print('----- Parsed output -----\n')
print(parsed)