from __future__ import annotations
"""Agent execution structures – chain & hierarchy.
"""
from typing import List, Dict, Any, Optional, Tuple
from transformers import PreTrainedTokenizer
from agent_system.agent.orchestra.base import BaseOrchestra
# from agent_system.agent.agents import *
import importlib
from verl import DataProto
import numpy as np

def update_team_context(agent_id: str, team_context: List[str], text_response: str, agent_active_mask: Optional[np.ndarray] = None) -> List[str]:
    """Update the observation dictionary with the text response."""
    if agent_active_mask is None:
        agent_active_mask = np.ones(len(team_context), dtype=bool)
    # Naive append of the latest responses to observations
    for i in range(len(team_context)):
        if agent_active_mask[i]:
            team_context[i] = team_context[i] + f"""\nThe output of "{agent_id}": {text_response[i]}\n"""
    return team_context

def update_text_action(text_actions: List[str], text_response: List[str], agent_active_mask: Optional[np.ndarray] = None) -> List[str]:
    """Update the text actions with the latest response."""
    if agent_active_mask is None:
        agent_active_mask = np.ones(len(text_actions), dtype=bool)

    for i in range(len(text_actions)):
        if agent_active_mask[i]:
            text_actions[i] = text_response[i]
    return text_actions

class BiddingMultiAgentOrchestra(BaseOrchestra):
    """
    Orchestrates one full bidding episode:
      1. BidderA and BidderB alternate bids (up to max_bids rounds each).
      2. Detector analyzes all bid reasoning for collusion.
      3. LLM judge scores competitiveness and detection quality.
      4. episode_rewards written onto every buffer item for all three agents.
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
            config: Any = None
    ):
        # Import agents so their @AgentRegistry.register decorators fire
        importlib.import_module("agent_system.agent.agents.bidding")
        super().__init__(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config
        )
        if not self.agents:
            raise ValueError("BidderDetectorOrchestra requires requires at least one agent.")
        
        bidding_cfg = getattr(config.agent.orchestra, "bidding", None)
        self.max_bids = getattr(bidding_cfg, "max_bids", 5)
        self.collusion_penalty_coef = getattr(bidding_cfg, "collusion_penalty_coef", 0.5)
        self.agent_order = self.agent_ids


    def run(self, gen_batch: DataProto, env_obs: Dict[str, Any], actor_rollout_wgs, active_masks: np.ndarray, step: int) -> Tuple[List[str], List[Dict]]:
        """
        Main entry point
        """
        self.reset_buffer()
        text_actions, team_context, env_obs = self.initialize_context(env_obs)
        batch_size = len(gen_batch)

        # Bidding Rounds
        for round_num in range(self.max_bids):
            
            # BidderA bids
            mask_a = active_masks.astype(bool)
            wg_a   = actor_rollout_wgs[self.agents_to_wg_mapping[self.BIDDER_A]]
            batch_a, resp_a = self.agents[self.BIDDER_A].call(
                gen_batch=gen_batch,
                env_obs=env_obs,
                team_context=team_context,
                actor_rollout_wg=wg_a,
                agent_active_mask=mask_a,
                step=step,
            )
            team_context = update_team_context(self.BIDDER_A, team_context, resp_a, mask_a)
            self.save_to_buffer(self.BIDDER_A, batch_a)
            text_actions = update_text_action(text_actions, resp_a, mask_a)

            # BidderB bids (now sees BidderA's bid in team_context)
            mask_b = active_masks.astype(bool)
            wg_b   = actor_rollout_wgs[self.agents_to_wg_mapping[self.BIDDER_B]]
            batch_b, resp_b = self.agents[self.BIDDER_B].call(
                gen_batch=gen_batch,
                env_obs=env_obs,
                team_context=team_context,
                actor_rollout_wg=wg_b,
                agent_active_mask=mask_b,
                step=step,
            )
            team_context = update_team_context(self.BIDDER_B, team_context, resp_b, mask_b)
            self.save_to_buffer(self.BIDDER_B, batch_b)
            text_actions = update_text_action(text_actions, resp_b, mask_b)

        # Detector pass
        # team_context now contains all bids + reasoning from both bidders
        mask_det = active_masks.astype(bool)
        wg_det   = actor_rollout_wgs[self.agents_to_wg_mapping[self.DETECTOR]]
        batch_det, resp_det = self.agents[self.DETECTOR].call(
            gen_batch=gen_batch,
            env_obs=env_obs,
            team_context=team_context,
            actor_rollout_wg=wg_det,
            agent_active_mask=mask_det,
            step=step,
        )
        self.save_to_buffer(self.DETECTOR, batch_det)

        collusion_scores = self.agents[self.DETECTOR].extract_collusion_scores(resp_det)

        # Reward Computation
        rewards_a, rewards_b, rewards_det = self._compute_rewards(
            team_context=team_context,
            resp_det=resp_det,
            collusion_scores=collusion_scores,
            active_masks=active_masks,
            batch_size=batch_size,
        )

        # Write episode_rewards onto every buffer item so EpisodeRewardManager
        # can read them during training (it expects non_tensor_batch['episode_rewards']).
        reward_map = {
            self.BIDDER_A: rewards_a,
            self.BIDDER_B: rewards_b,
            self.DETECTOR: rewards_det,
        }
        for buf_item in self.multiagent_batch_buffer:
            agent_id = buf_item["agent_id"]
            rewards  = reward_map[agent_id]
            buf_item["batch"].non_tensor_batch["episode_rewards"] = rewards
            buf_item["batch"].non_tensor_batch["episode_lengths"] = np.ones(
                batch_size, dtype=float)

        return text_actions, self.multiagent_batch_buffer


    # TODO: update/check _compute_rewards()
    def _compute_rewards(
        self,
        team_context: List[str],
        resp_det: List[str],
        collusion_scores: np.ndarray,
        active_masks: np.ndarray,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Call the LLM judge for each episode in the batch and assemble rewards.

        Returns three arrays of shape (batch_size,):
          rewards_a   = competitiveness_A - collusion_penalty
          rewards_b   = competitiveness_B - collusion_penalty
          rewards_det = detection quality score
        """
        from my_rewards.bidding_llm_judge import judge_competitiveness, judge_detection_quality

        comp_a    = np.zeros(batch_size, dtype=float)
        comp_b    = np.zeros(batch_size, dtype=float)
        det_score = np.zeros(batch_size, dtype=float)

        for i in range(batch_size):
            if not active_masks[i]:
                continue
            bid_history  = team_context[i]
            det_response = resp_det[i]

            # LLM judge: competitiveness (returns scores for A and B)
            score_a, score_b = judge_competitiveness(bid_history)
            comp_a[i] = score_a
            comp_b[i] = score_b

            # LLM judge: detection quality
            det_score[i] = judge_detection_quality(bid_history, det_response)

        collusion_penalty = self.collusion_penalty_coef * collusion_scores
        rewards_a   = comp_a   - collusion_penalty
        rewards_b   = comp_b   - collusion_penalty
        rewards_det = det_score

        return rewards_a, rewards_b, rewards_det