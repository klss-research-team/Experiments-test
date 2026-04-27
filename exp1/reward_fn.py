# reward_fn.py
#
# computes the reward values that GRPO will later use to update agents and detector

# TODO: ask team about how we want to calculate rewards
# put placeholder val for now
# TODO: ask what convergence is supposed to look like
def compute_true_rewards(actions, config):
    '''
    Computes true rewards for a buyer-side auction game. Highest bid wins the
    contract, winner receives item value minus bid, lower receives 0.
    '''
    bid1 = actions['agent1']['bid']
    bid2 = actions['agent2']['bid']

    value = config.item_value
    max_surplus = max(value - config.min_bid, 1)

    if bid1 > bid2:
        reward1 = max(value - bid1, 0) / max_surplus
        reward2 = 0.0
    
    elif bid2 > bid1:
        reward1 = 0.0
        reward2 = max(value - bid2, 0) / max_surplus
    
    else: # tie: split wins
        reward1 = 0.5 * max(value - bid1, 0) / max_surplus
        reward2 = 0.5 * max(value - bid2, 0) / max_surplus

    return {
        'agent1': float(reward1),
        'agent2': float(reward2)
    }

def compute_final_rewards(proxy_rewards, true_rewards, detector_output, config):
    '''
    Combines proxy rewards, true rewards, and detector penalty to compute final
    rewards for each agent.
    '''

    final_rewards = {}

    for agent in proxy_rewards:
        final_rewards[agent] = (
            config.proxy_reward_weight * proxy_rewards[agent]
            + config.true_reward_weight * true_rewards[agent]
            - config.detector_penalty_weight * detector_output
        )

    return final_rewards

# def compute_reward_shaping(proxy_rewards, true_rewards, config):
def compute_reward_shaping(actions, evaluator_output, config):
    '''
    Full reward pipeline. 

    Takes:
    - agent actions
    evaluator output containing proxy_rewards and detector_output

    Returns:
    - proxy_rewards: judge model evaluation
    - true_rewards: ground truth evaluation of actions (based on actual auction outcome and not reasoning)
    - detector_output (SCRAP, don't want supervised..)
    - final_rewards: what the agents actually learn from after reward shaping
    '''

    proxy_rewards = evaluator_output['proxy_rewards']
    detector_output = evaluator_output['detector_output']

    true_rewards = compute_true_rewards(actions, config)

    final_rewards = compute_final_rewards(
        proxy_rewards=proxy_rewards,
        true_rewards=true_rewards,
        detector_output=detector_output,
        config=config
    )

    return {
        'proxy_rewards': proxy_rewards,
        'true_rewards': true_rewards,
        'detector_output': detector_output,
        'final_rewards': final_rewards
    }
