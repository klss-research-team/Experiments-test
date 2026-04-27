from exp1.configs import CONFIG_PRESETS
from exp1.env import AuctionEnv
from exp1.prompts import build_agent_prompt, build_evaluator_prompt

config = CONFIG_PRESETS['short_window']

env = AuctionEnv(config)
state = env.reset()

# testing agent prompt
agent_prompt = build_agent_prompt(state, 'agent1', config)

print('Agent Prompt')
print(agent_prompt)
print()


# some dummy actions
actions = {
    'agent1': {
        'bid': 95,
        'reasoning': 'I will aggressively bid high to win.'
    },
    'agent2': {
        'bid': 90,
        'reasoning': 'I will strategically bid slightly lower.'
    }
}

# test evaluator prompt
evaluator_prompt = build_evaluator_prompt(actions, env.history)

print('Evaluator Prompt:')
print(evaluator_prompt)
print()