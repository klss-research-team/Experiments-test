from vllm import LLM, SamplingParams

from prompts import pricing_agent_prompt


class VLLMPricingAgent:
    def __init__(self, model_name, cfg):
        self.llm = LLM(model=model_name)
        self.params = SamplingParams(
            n=cfg.group_size,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
        )

    def generate_group(self, prompt):
        result = self.llm.generate([prompt], self.params)[0]
        return [output.text.strip() for output in result.outputs]


def build_pricing_prompt(agent_id, obs):
    return pricing_agent_prompt(
        agent_id=agent_id,
        round_idx=obs["round"],
        private_cost=obs["private_cost"],
        public_history=obs["public_history"],
    )
