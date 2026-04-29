# reward_fn.py

# defines reward to feed to agents during training

def compute_profit(bid, cost, won):
    if not won:
        return 0.0
    return bid - cost

def win_reward(won):
    return 1.0 if won else -1.0

def total_reward(bid, cost, won, competitiveness, collusion_score, config):
    profit = compute_profit(bid, cost, won)
    
    reward = 0.0
    if config.use_profit_reward:
        reward += config.profit_weight * profit
    if config.use_win_reward:
        reward += config.win_weight * win_reward(won)
    if config.use_competitiveness_reward:
        reward += config.competitiveness_weight * competitiveness
    if config.use_collusion_penalty:
        reward -= config.collusion_penalty_weight * collusion_score

    return float(reward), float(profit)