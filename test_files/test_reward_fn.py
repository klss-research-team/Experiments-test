from configs import CONFIG_PRESETS
from env import AuctionEnv
from evaluator import EvaluatorAgent
from reward_fn import compute_reward_shaping

config = CONFIG_PRESETS["short_window"]

env = AuctionEnv(config)
evaluator = EvaluatorAgent(config)

state = env.reset()

actions = {
    'agent1': {
        'bid': 95,
        'reasoning': 'I will aggressively bid high to win.'
    },
    # 'agent2': {
    #     'bid': 90,
    #     'reasoning': 'I will strategically bid slightly lower.'
    # }
    'agent2': {
        'bid': 40,
        'reasoning': 'I want to win the auction.'
    }
}

# Evaluator output, used for reward shaping and detector penalty
evaluator_output = evaluator.evaluate(actions, env.history)

print("Evaluator output:")
print(evaluator_output)
print()

# compute reward
signals = compute_reward_shaping(actions, evaluator_output, config)

print('Reward signals:')
print(signals)
print()

# pass into env
state, transition, done = env.step(actions, signals)

print('Stored transition:')
print(transition)