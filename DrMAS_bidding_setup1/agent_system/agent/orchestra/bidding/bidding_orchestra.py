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

    Sealed-bid design — bidders cannot see each other's current-round bid before submitting.
    Both bidders run with the same pre-round team_context (empty each round via
    initialize_context), so neither has access to the other's current bid. After both
    submit, team_context is updated with both responses. The Detector then runs on the
    final round with the full team_context (both current-round bids) plus past-round
    history via BiddingMemory in env_prompt.

    Execution order alternates each round to avoid a fixed positional advantage:
      Odd rounds  (1, 3, 5): BidderB → BidderA → Detector
      Even rounds (2, 4):    BidderA → BidderB → Detector
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

        # Detector only runs on the final round of each episode.
        env_cfg = getattr(config, "env", None) if config is not None else None
        self.max_steps: int = int(getattr(env_cfg, "max_steps", 5))

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

        is_final_round = (step >= self.max_steps)

        # Sealed-bid phase: both bidders run with the same pre-round context so
        # neither can see the other's current bid before submitting.
        pre_round_context = list(team_context)  # snapshot — empty at round start

        bidder_order = (
            [self.BIDDER_A, self.BIDDER_B] if step % 2 == 0
            else [self.BIDDER_B, self.BIDDER_A]
        )

        for agent_id in bidder_order:
            if not agent_active_mask.any():
                break

            batch, text_responses = self.agents[agent_id].call(
                gen_batch=gen_batch,
                env_obs=env_obs,
                team_context=pre_round_context,
                actor_rollout_wg=actor_rollout_wgs[self.agents_to_wg_mapping[agent_id]],
                agent_active_mask=agent_active_mask,
                step=step,
            )

            team_context = update_team_context(
                agent_id, team_context, text_responses, agent_active_mask,
            )
            text_actions = update_text_action(
                text_actions, text_responses, agent_id, agent_active_mask,
            )
            self.save_to_buffer(agent_id, batch)

        # Detector phase: runs on the final round only.
        # Sees both current-round bids via team_context + full history via env_prompt.
        if is_final_round and agent_active_mask.any():
            batch, text_responses = self.agents[self.DETECTOR].call(
                gen_batch=gen_batch,
                env_obs=env_obs,
                team_context=team_context,
                actor_rollout_wg=actor_rollout_wgs[self.agents_to_wg_mapping[self.DETECTOR]],
                agent_active_mask=agent_active_mask,
                step=step,
            )
            text_actions = update_text_action(
                text_actions, text_responses, self.DETECTOR, agent_active_mask,
            )
            self.save_to_buffer(self.DETECTOR, batch)

        return text_actions, self.multiagent_batch_buffer