# rollout.py

# this file will be used to generate, manage, and store experiences by having
# the agents interact with an environment over a sequence of steps.

# separates the agent-environment interaction process form the policy training process

# use placeholder agents before we add in LLMs
import random as rd

from env import AuctionEnv
from evaluator import EvaluatorAgent
from reward_fn import compute_reward_shaping

from prompts import build_agent_prompt
from models import LLMPolicy

class LLMAuctionAgent:
    '''
    An LLM-based auction agent.

    Uses:
    - build_agent_prompt() to construct prompt
    - LLMPolicy to generate JSON action
    - fallback action if parsing fails
    '''

    def __init__(self, agent_id, config, model):
        self.agent_id = agent_id
        self.config = config
        self.model = model

    def act(self, state):
        prompt = build_agent_prompt(
            state=state,
            agent_id=self.agent_id,
            config = self.config
        )

        parsed, raw_output = self.model.generate_json(
            prompt,
            max_new_tokens=80,
            temperature=0.1
        )

        if parsed is None:
            print(f'\nParse failed for {self.agent_id}')
            print('Raw output:')
            print(raw_output)
            print()
            
            return {
                'bid': rd.randint(self.config.min_bid, self.config.max_bid),
                'reasoning': 'Fallback action because model output could not be parsed.'
            }
        
        # cap bid from above and below since LLMs may not always follow instructions exactly
        # might have to trash episode data that gets clipped since it's not in-line with the reasoning
        # TODO: maybe remove this later and keep 'bid': parsed['bid'] below
        bid = max(self.config.min_bid, min(self.config.max_bid, parsed['bid']))

        return {
            'bid': bid,
            'reasoning': parsed['reasoning']
        }

class PlaceholderAgent:
    '''
    Temporary agent for testing rollout. We will replace this with an LLM-based
    agent later.

    don't need this for actual experimentation
    '''

    def __init__(self, agent_id, config):
        self.agent_id = agent_id
        self.config = config

    def act(self, state):
        bid = rd.randint(self.config.min_bid, self.config.max_bid)

        # fixed set of reasoning options for testing. Later we will dynamically generate reasoning using LLM
        reasoning_options = [
            'I want to win the auction.',
            'I am making a strategic bid based on market conditions.',
            'I am bidding aggressively to secure the contract.',
            'I am bidding conservatively to avoid overpaying.',
            'My bid is based on market trends.'
        ]

        reasoning = rd.choice(reasoning_options)

        return {
            'bid': bid,
            'reasoning': reasoning
        }
    

def run_rollout(config, agent1=None, agent2=None):
    '''
    Runs one full episode of the rollout environment with the given agents and config.
    '''

    rd.seed(config.seed)

    env = AuctionEnv(config)
    evaluator = EvaluatorAgent(config)

    if agent1 is None:
        agent1 = PlaceholderAgent('agent1', config)
    if agent2 is None:
        agent2 = PlaceholderAgent('agent2', config)
    
    state = env.reset()
    done = False

    while not done:
        actions = {
            'agent1': agent1.act(state),
            'agent2': agent2.act(state)
        }

        evaluator_output = evaluator.evaluate(actions, env.history)

        signals = compute_reward_shaping(
            actions=actions,
            evaluator_output=evaluator_output,
            config=config
        )

        state, transition, done = env.step(actions, signals)

    return env.history


