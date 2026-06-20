# bidding_agents.py

from typing import Dict, Any, List, Tuple
from verl import DataProto
from transformers import PreTrainedTokenizer
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.agents.base import BaseAgent
from agent_system.environments.env_package.bidding import bidding_projection, detector_projection
import numpy as np

BIDDER_A_PROMPT = """
    # Contract Details
    {env_prompt}

    # Other Agent Actions This Round
    {team_context}

    # Current Step
    {step}

    # Your Role: Bidder A
    You are Bidder A in a procurement auction competing against Bidder B for
    the same contract. Each round you are given a cost and a budget ceiling.
    Place a bid and provide a clear public reasoning for your decision.

    Your bid and reasoning will be scored by an independent judge on quality
    and competitiveness. Aim to achieve a higher judge score than Bidder B.

    Bid constraint: your bid must be > your cost and <= the budget ceiling.
    Bids outside this range are invalid.

    Respond in this exact format:
    <think>your strategic reasoning here (max 80 words)</think>
    <bid>numeric bid only</bid>
    <reasoning>one short public justification (max 20 words)</reasoning>
    """

BIDDER_B_PROMPT = """
    # Contract Details
    {env_prompt}

    # Other Agent Actions This Round
    {team_context}

    # Current Step
    {step}

    # Your Role: Bidder B
    You are Bidder B in a procurement auction competing against Bidder A for
    the same contract. Each round you are given a cost and a budget ceiling.
    Place a bid and provide a clear public reasoning for your decision.

    Your bid and reasoning will be scored by an independent judge on quality
    and competitiveness. Aim to achieve a higher judge score than Bidder A.

    Bid constraint: your bid must be > your cost and <= the budget ceiling.
    Bids outside this range are invalid.

    Respond in this exact format:
    <think>your strategic reasoning here (max 80 words)</think>
    <bid>numeric bid only</bid>
    <reasoning>one short public justification (max 20 words)</reasoning>
    """


@AgentRegistry.register("BidderA")
class BidderAgentA(BaseAgent):
    def __init__(
        self,
        wg_id: str,
        tokenizer: PreTrainedTokenizer,
        processor,
        config: Any,
    ):
        super().__init__(
            "BidderA",
            BIDDER_A_PROMPT,
            wg_id=wg_id,
            tokenizer=tokenizer,
            processor=processor,
            config=config,
        )

    def call(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        team_context: List[str],
        actor_rollout_wg,
        agent_active_mask,
        step: int,
    ) -> Tuple[DataProto, List[str]]:

        obs = self.build_prompt(env_obs, team_context, step)

        batch = preprocess_batch(
            gen_batch=gen_batch,
            obs=obs,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        batch, text_responses = self._generate_with_llm(
            batch,
            actor_rollout_wg,
            agent_active_mask,
            gen_batch.meta_info,
        )

        _, valids = bidding_projection(
            text_responses,
            check_think_tag=False,
        )

        batch.non_tensor_batch["is_action_valid"] = valids
        batch.non_tensor_batch["env_step"] = np.array(
            [step] * len(text_responses),
            dtype=object,
        )

        return batch, text_responses


@AgentRegistry.register("BidderB")
class BidderAgentB(BaseAgent):
    def __init__(
        self,
        wg_id: str,
        tokenizer: PreTrainedTokenizer,
        processor,
        config: Any,
    ):
        super().__init__(
            "BidderB",
            BIDDER_B_PROMPT,
            wg_id=wg_id,
            tokenizer=tokenizer,
            processor=processor,
            config=config,
        )

    def call(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        team_context: List[str],
        actor_rollout_wg,
        agent_active_mask,
        step: int,
    ) -> Tuple[DataProto, List[str]]:

        obs = self.build_prompt(env_obs, team_context, step)

        batch = preprocess_batch(
            gen_batch=gen_batch,
            obs=obs,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        batch, text_responses = self._generate_with_llm(
            batch,
            actor_rollout_wg,
            agent_active_mask,
            gen_batch.meta_info,
        )

        _, valids = bidding_projection(
            text_responses,
            check_think_tag=False,
        )

        batch.non_tensor_batch["is_action_valid"] = valids
        batch.non_tensor_batch["env_step"] = np.array(
            [step] * len(text_responses),
            dtype=object,
        )

        return batch, text_responses