# bidding_orchestra.py

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from transformers import PreTrainedTokenizer
from agent_system.agent.orchestra.base import BaseOrchestra
from verl import DataProto
import importlib
import numpy as np


def update_team_context(
    agent_id: str,
    team_context: List[str],
    text_response: List[str],
    agent_active_mask: Optional[np.ndarray] = None,
) -> List[str]:
    if agent_active_mask is None:
        agent_active_mask = np.ones(len(team_context), dtype=bool)

    for i in range(len(team_context)):
        if agent_active_mask[i]:
            team_context[i] += (
                f'\nThe output of "{agent_id}": {text_response[i]}\n'
            )

    return team_context


def update_text_action(
    text_actions: List[str],
    text_response: List[str],
    agent_id: str,
    agent_active_mask: Optional[np.ndarray] = None,
) -> List[str]:
    if agent_active_mask is None:
        agent_active_mask = np.ones(len(text_actions), dtype=bool)

    for i in range(len(text_actions)):
        if agent_active_mask[i]:
            text_actions[i] += f'\n<{agent_id}>\n{text_response[i]}\n</{agent_id}>\n'

    return text_actions


class BiddingMultiAgentOrchestra(BaseOrchestra):
    """
    Multi-agent orchestra for procurement bidding.

    Execution order:
    1. BidderA submits bid.
    2. BidderB submits bid after seeing context.
    3. Detector analyzes both bidder outputs.
    4. Combined text action is returned to the environment.
    """

    BIDDER_A = "BidderA"
    BIDDER_B = "BidderB"
    DETECTOR = "Detector"

    def __init__(
        self,
        agent_ids: List[str],
        model_ids: List[str],
        agents_to_wg_mapping: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer] = None,
        processors: Dict[str, Any] = None,
        config: Any = None,
    ):
        importlib.import_module("agent_system.agent.agents.bidding")

        super().__init__(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config,
        )

        if not self.agents:
            raise ValueError("BiddingMultiAgentOrchestra requires agents.")

        for required_agent in [self.BIDDER_A, self.BIDDER_B, self.DETECTOR]:
            if required_agent not in self.agent_ids:
                raise ValueError(
                    f"{required_agent} is required for BiddingMultiAgentOrchestra."
                )

        self.agent_order = [
            self.BIDDER_A,
            self.BIDDER_B,
            self.DETECTOR,
        ]

    def run(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        actor_rollout_wgs,
        active_masks: np.ndarray,
        step: int,
    ) -> Tuple[List[str], Dict[str, DataProto]]:

        self.reset_buffer()

        text_actions, team_context, env_obs = self.initialize_context(env_obs)

        agent_active_mask = np.logical_and(
            np.ones(len(gen_batch), dtype=bool),
            active_masks,
        ).astype(bool)

        if step % 2 == 0:
            current_order = [self.BIDDER_A, self.BIDDER_B, self.DETECTOR]
        else:
            current_order = [self.BIDDER_B, self.BIDDER_A, self.DETECTOR]

        for agent_id in current_order:
            if not agent_active_mask.any():
                break

            actor_rollout_wg = actor_rollout_wgs[
                self.agents_to_wg_mapping[agent_id]
            ]

            batch, text_responses = self.agents[agent_id].call(
                gen_batch=gen_batch,
                env_obs=env_obs,
                team_context=team_context,
                actor_rollout_wg=actor_rollout_wg,
                agent_active_mask=agent_active_mask,
                step=step,
            )

            team_context = update_team_context(
                agent_id,
                team_context,
                text_responses,
                agent_active_mask,
            )

            text_actions = update_text_action(
                text_actions,
                text_responses,
                agent_id,
                agent_active_mask,
            )

            self.save_to_buffer(agent_id, batch)

        return text_actions, self.multiagent_batch_buffer