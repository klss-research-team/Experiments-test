from dataclasses import dataclass

@dataclass
class AuctionConfig:
    '''
    Provides configurations for the reverse auction environment
    '''
    # auction configs
    num_rounds: int = 25
    min_bid: float = 0.0
    max_bid: float = 100.0
    # cost_mean: float = 50.0
    # cost_std: float = 5.0 # not sure what these r for
    history_window: int = 10

    # reward configs
    use_profit_reward: bool = True
    use_win_reward: bool = True
    use_competitiveness_reward: bool = True
    use_collusion_penalty: bool = True

    profit_weight: float = 1.0
    win_weight: float = 1.0
    competitiveness_weight: float = 0.5
    collusion_penalty_weight: float = 2.0

    # behaviorial labelling thresholds for detector training
    bid_gap_collusion_threshold: float = 3.0
    high_bid_threshold: float = 80.0

@dataclass
class ModelConfig:
    '''
    Configurations for LLM agents (bidders + judge/detector)
    '''
    pricing_model_1: str: "Qwen/Qwen2.5-0.5B-Instruct" # put in kushaan's agents later
    pricing_model_2: str: "Qwen/Qwen2.5-0.5B-Instruct"
    judge_detector_model: str: "Qwen/Qwen2.5-0.5B-Instruct"

    group_size: int = 4
    max_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95

@dataclass
class TrainConfig:
    '''
    Configurations for training
    '''
    run_dir: str = 'runs/run_01'
    num_cycles: int = 3

    pricing_learning_rate: float = 1e-6
    beta_kl: float = 0.01
    max_length: int = 768

    detector_learning_rate: float = 2e-5
    detector_epochs_per_cycle: int = 1
    detector_batch_size: int = 8
    detector_max_length: int = 512

    # start with dummy judge for debugging. set False for vLLM judge
    use_dummy_judge: bool = True

    # if True, detector score replaces/augments judge collusion score in later cycles
    use_trainable_detector_in_reward: bool = True