# pricing_agent.py
#
# abstraction for pricing agent model
# 
# - generative LLM policy
# - generates reasoning + bid
# - trained with GRPO

from vllm import LLM, SamplingParams
from prompts import pricing_agent_prompt


class PricingAgent:
    def __init__(self, model_path, cfg):
        self.model_path = model_path
        self.llm = LLM(model=model_path)

        self.params = SamplingParams(
            n=cfg.group_size,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
        )

    def build_prompt(self, agent_id, obs):
        return pricing_agent_prompt(
            agent_id=agent_id,
            round_idx=obs["round"],
            private_cost=obs["private_cost"],
            public_history=obs["public_history"],
        )

    def generate_group(self, agent_id, obs):
        prompt = self.build_prompt(agent_id, obs)
        result = self.llm.generate([prompt], self.params)[0]
        responses = [out.text.strip() for out in result.outputs]
        return prompt, responses