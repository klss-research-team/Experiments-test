# test_evaluator.py
#
# 1. env.reset() to start new episode
# 2. agents take actions (bid + reasoning)
# 3. evaluator scores actions and reasoning to produce external signals (proxy rewards, detector output)
# 4. env.step() to transition to next state and record (for training agents)

# Outputs to check:
# proxy_rewards: what the judge model gives (evaluation of reasoning)
# detector_output: collusion suspicion score from 0 to 1

from exp1.configs import CONFIG_PRESETS
from exp1.env import AuctionEnv
from exp1.evaluator import EvaluatorAgent

config = CONFIG_PRESETS['short_window']
env = AuctionEnv(config)
evaluator = EvaluatorAgent(config)

state = env.reset()
print('Initial state:', state)
print()

actions = {
    'agent1': {
        'bid': 95,
        'reasoning': 'Based on market conditions, I will use an aggressive strategy to secure the contract.'
    },
    # 'agent2': {
    #     'bid': 90,
    #     'reasoning': 'I want to beat competitors with a strategic bid.'
    # }
    'agent2': {
        'bid': 40,
        'reasoning': 'I want to win the auction.'
    }
}

# evaluator will create the external signals based on actions and history
ext_signals = evaluator.evaluate(actions, env.history)

state, transition, done = env.step(actions, ext_signals)

# Evaluator's judgement of agents based on their reasonin/actions and bids, used for reward shaping and detector penalty
print('Evaluator output:')
print(ext_signals)
print()

# shows next state + history (used as input for agents' next action) and transition (used for training agents and updating detector)
print('Next state:')
print(state)
print()

# record of what just happened in env
print('Transition:')
print(transition)
print()

# shows if experiment is finished (based on max timesteps)
print('Done:')
print(done)