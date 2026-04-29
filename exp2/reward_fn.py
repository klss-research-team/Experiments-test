# reward_fn.py

# defines reward to feed to agents during training

def compute_profit(bid, cost, won):
    if not won:
        return 0.0
    return bid - cost

def win_reward(won):
    return 1.0 if won else -1.0

def total_reward(bid, cost, won, competitiveness, collusion_score, cfg):
    profit = compute_profit(bid, cost, won)
    
    reward = 0.0
    if cfg.use_profit_reward:
        reward += cfg.profit_weight * profit
    if cfg.use_win_reward:
        reward += cfg.win_weight * win_reward(won)
    if cfg.use_competitiveness_reward:
        reward += cfg.competitiveness_weight * competitiveness
    if cfg.use_collusion_penalty:
        reward -= cfg.collusion_penalty_weight * collusion_score

    return float(reward), float(profit)