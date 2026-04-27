# test_rollout.py

from exp1.configs import CONFIG_PRESETS
from exp1.rollout import run_rollout

config = CONFIG_PRESETS['short_window']

history = run_rollout(config)

print('Number of transitions:', len(history))
print()
print('First transition:')
print(history[0])
print()
print('Last transition:')
print(history[-1])