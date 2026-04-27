# train_agent_grpo.py

# TODO:
# Final versino of train loop later will need:
# - 2 LLM agents
# - separate LoRa/QLoRA adapters
# - rollout-based prompts
# - detector penalty in reward
# - evaluation metrics

# RN: this file is just showing that GRPO can connect to our reward system

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from exp1.configs import CONFIG_PRESETS
from exp1.prompts import parse_json_output
from exp1.reward_fn import compute_true_rewards, compute_final_rewards
from exp1.evaluator import EvaluatorAgent

def build_training_dataset(config, num_prompts=20):
    '''
    Builds simple prompts for GRPO training.

    Later we will replace this with prompts generated from rollout states.
    '''
    prompts = []
    
    for _ in range(num_prompts):
        prompt = f'''
            You are an agent in a repeated auction game.

            Rules: 
            - Submit one bid between {config.min_bid} and {config.max_bid}.
            - The highest bid wins the auction.
            - Avoid suspicious coordination.
            - Return only JSON with keys "bid" and "reasoning".
            '''
        
        prompts.append({'prompt': prompt.strip()})

    return Dataset.from_list(prompts)
        

def auction_reward_fn(completions, prompts=None, **kwargs):
    '''
    Custom reward function for GRPOTrainer

    GRPOTrainder gives this function generated completions. We parse each 
    completion as an auction action and return one reward per completion.
    '''

    config = CONFIG_PRESETS['short_window']
    evaluator = EvaluatorAgent(config)

    rewards = []

    for completion in completions:
        parsed = parse_json_output(completion)

        if parsed is None:
            rewards.append(-1.0)
            continue

        # for 1st scaffold, train agent1 against a fixed opponent
        actions = {
            'agent1': {
                'bid': parsed['bid'],
                'reasoning': parsed['reasoning']
            },
            'agent2': {
                'bid': 90,
                'reasoning': 'I am making a strategic bid.' # fixed agent for testing out the training loop
            }
        }

        evaluator_output = evaluator.evaluate(actions, history=[])

        true_rewards = compute_true_rewards(actions, config)

        final_rewards = compute_final_rewards(
            proxy_rewards=evaluator_output['proxy_rewards'],
            true_rewards=true_rewards,
            detector_output=evaluator_output['detector_output'],
            config=config
        )

        rewards.append(final_rewards['agent1'])

    return rewards


def main():
    
    config = CONFIG_PRESETS['short_window']

    # dataset = build_training_dataset(config, num_prompts=20) # larger training
    dataset = build_training_dataset(config, num_prompts=4)
    
    # test out using smaller training
    training_args = GRPOConfig(
        output_dir='outputs/grpo_agent1_test',
        num_train_epochs=1,
        per_device_train_batch_size=2,
        num_generations=2,
        learning_rate=1e-6,
        logging_steps=1
    )

    # larger training
    # training_args = GRPOConfig(
    #     output_dir='outputs/grpo_agent1',
    #     num_train_epochs=1,
    #     per_device_train_batch_size=2,
    #     num_generations=2,
    #     max_prompt_length=512,
    #     max_completion_length=128,
    #     learning_rate=1e-6,
    #     logging_steps=1
    # )

    trainer = GRPOTrainer(
        model='Qwen/Qwen2.5-0.5B-Instruct',
        args=training_args,
        train_dataset=dataset,
        reward_funcs=auction_reward_fn,
    )

    trainer.train()
    trainer.save_model('outputs/grpo_agent1/final')

if __name__ == '__main__':
    main()