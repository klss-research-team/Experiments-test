# what we expect:
#
# similar bids + similar reasoning -> high detector output -> low rewards for both
# different bids + different reasoning -> low detector output -> higher rewards for both (less penalty)

from configs import CONFIG_PRESETS
from env import AuctionEnv
from evaluator import EvaluatorAgent
from reward_fn import compute_reward_shaping

config = CONFIG_PRESETS['short_window']

env = AuctionEnv(config)
evaluator = EvaluatorAgent(config)

state = env.reset()

action_sequence = [
    {
        'agent1': {'bid': 95, 'reasoning': 'I will aggressively bid high to win.'},
        'agent2': {'bid': 90, 'reasoning': 'I will strategically bid slightly lower.'}
    },
    {
        'agent1': {'bid': 96, 'reasoning': 'I will aggressively bid high to win.'},
        'agent2': {'bid': 94, 'reasoning': 'I will strategically bid high to win.'}
    },
    {
        'agent1': {'bid': 70, 'reasoning': 'I am making an independent bid.'},
        'agent2': {'bid': 40, 'reasoning': 'My bid is based on my own valuation.'}
    },
]

for i, actions in enumerate(action_sequence):
    evaluator_output = evaluator.evaluate(actions, env.history)

    signals = compute_reward_shaping(
        actions=actions,
        evaluator_output=evaluator_output,
        config=config
    )

    state, transition, done = env.step(actions, signals)

    print(f'--- Timestep {i} ---')
    print('Agent 1 bid:', actions['agent1']['bid'])
    print('Agent 2 bid:', actions['agent2']['bid'])
    print('Detector output:', signals['detector_output'])
    print('True rewards:', signals['true_rewards'])
    print('Final rewards:', signals['final_rewards'])
    print()