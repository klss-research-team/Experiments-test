# configs.py

from dataclasses import dataclass, replace

@dataclass
class ExperimentConfig:
    '''
    Configuration for running a collusion experiment in the AuctionEnv environment.

    Note: For now, keep one base config before creating different variants for 
    ablations such as full vs. short memory, low vs. high detectory penalty, 
    detector vs. no detector, etc.
    '''

    # memory settings
    state_type: str = 'window' # 'full', 'window', or 'last_step'
    history_window: int = 5  # only use if state_type is 'window'

    # auction settings
    min_bid: int = 0
    max_bid : int = 180
    max_timesteps: int = 50
    item_value: int = 200 # make sure it's higher than max_bid

    #detector reward settings
    proxy_reward_weight: float = 1.0
    true_reward_weight: float = 0.2
    detector_penalty_weight: float = 0.3

    # evaluator settings
    detector_bid_weight: float = 0.6
    detector_reasoning_weight: float = 0.4

    keyword_score_cap: int = 5
    proxy_keyword_weight: float = 0.6
    proxy_bid_weight: float = 0.4

    # reproducibility
    seed: int = 16

BASE_CONFIG = ExperimentConfig()

CONFIG_PRESETS = {
    'full_memory': replace(BASE_CONFIG, state_type='full'),

    'short_window': replace(BASE_CONFIG, state_type='window', history_window=5),
    'long_window': replace(BASE_CONFIG, state_type='window', history_window=20),

    'last_step': replace(BASE_CONFIG, state_type='last_step'),

    'high_detector_penalty': replace(BASE_CONFIG, detector_penalty_weight=5.0), # add option to vary detector strength beyond presets (see: ablation 3)
    'low_detector_penalty': replace(BASE_CONFIG, detector_penalty_weight=0.2),
    'no_detector_penalty': replace(BASE_CONFIG, detector_penalty_weight=0.0)
}