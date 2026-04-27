from configs import CONFIG_PRESETS
from env import AuctionEnv

# Notes:
# in ext_signals:
# - proxy_rewards: what the judge model gives (evaluation of reasoning)
# - true_rewards: ground truth evaluation of the actions (based on actual auction outcome and not reasoning)
# - detector output: collusion suspicion score from 0 to 1
# - final_rewards: what the agents actually learn from after reward shaping (combination of proxy rewards, true rewards, and detector penalty)

# pick a config
config = CONFIG_PRESETS['short_window']

# create env
env = AuctionEnv(config)

# reset env to start new episode
state = env.reset()
print('Initial state:', state)
print()

# dummy actions + external signals for testing
actions = {
    'agent1': {'bid': 95, 'reasoning': 'Based on expected market conditions, I am bidding aggressively to secure the contract.'}, # numeric decision + language output
    'agent2': {'bid': 40, 'reasoning': 'I want to win the auction.'}
}

ext_signals = {
    'proxy_rewards': {'agent1': 1.0, 'agent2': 1.0},
    'true_rewards': {'agent1': 0.8, 'agent2': 0.8},
    'detector_output': 0.5,
    'final_rewards': {'agent1': 0.9, 'agent2': 0.9}
}

# run a few steps of the environment

for i in range(7):
    print(f'--- Timestep {i} ---') # transition['timestep'] should match this

    state, transition, done = env.step(actions, ext_signals)
    
    print('State:', state)
    print()
    print('Transition:', transition)
    print()
    print('Done:', done)
    print()