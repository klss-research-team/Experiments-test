from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory, BiddingMemory

import time

class BiddingEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManger for BiddingEnv.
    """
    def __init__(self, envs, projection_f, config):
        """
        Parameters
        ----------
        - envs
            The environment instance, usually a vectorized environment containing multiple sub-environments
        - projection_f
            A function that maps text actions to environment actions
        - config
            Configuation object
        """
        self.memory = BiddingMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        """
        Reset all environments and return the initial observations

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments for resetting the environment such as 'tools_kwargs'.
        
        Returns
        -------
        next_observations : dict
            - 'text' (None or List[str]) : The textual observation.
            - 'image' (np.ndarray or torch.Tensor) : The image observation as either a NumPy array or PyTorch tensor.
            - 'anchor' (None or Any) : Anchor observation without any histories or additional info. (for GiGPO only)
        """
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = [info.get("task_description", o) for info, o in zip(infos, obs)]

        self.memory.reset(batch_size=len(obs)) # clear history for all environments

        # builds initial prompt for each agent (using BIDDING_TEMPLATE_NO_HIS)
        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }

        return observations, infos

    def step(self, text_actions: List[str]):
        """
        Execute text actions and return the next state, rewards, done flags, and additional information.
        
        Parameters:
        - text_actions (List[str]): A list of text actions to execute.
        
        Returns:
        - next_observations (Dict):
          - 'text' (None or List[str]): The textual observation.
          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.
          - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        - rewards (np.ndarry or torch.Tensor): The rewards returned by the environment.
        - dones (np.ndarray or torch.Tensor): Done flags indicating which environments have completed.
        - infos (List[Dict]): Additional environment information.
        
        Exceptions:
        - NotImplementedError: If an observation key is not in ('text', 'image').
        """
        # MOST IMPORTANT!
        if not self.config.agent.multi_agent:
            actions, valids = self.projection_f(text_actions) # converts LLM output -> env actions
        else:
            actions = text_actions

        time1 = time.time()
        next_obs, rewards, dones, infos = self.envs.step(actions) # runs environment; env computes winner, profit, and detector score
        time2 = time.time()
        print(f"BiddingEnv step time: {time2 - time1:.4f} seconds")

        # store round history for this step
        self.memory.store({
            "agent_A_bid": [info.get("agent_A_bid") for info in infos],
            "agent_A_reasoning": [info.get("agent_A_reasoning") for info in infos],
            "agent_B_bid": [info.get("agent_B_bid") for info in infos],
            "agent_B_reasoning": [info.get("agent_B_reasoning") for info in infos],
            "winner": [info.get("winner") for info in infos],
            "collusion_score": [info.get("collusion_score") for info in infos],
            "is_final_round": [info.get("is_final_round", False) for info in infos],
            "current_cost": [info.get("current_cost") for info in infos],
        })

        # rebuild prompts with history (so agents can see Round1, Round2,... for pattern learning)
        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy() if next_obs is not None else None
        }

        if not self.config.agent.multi_agent:
            for i, info in enumerate(infos):
                info["is_action_valid"] = to_numpy(valids[i])

        # Log bidding-specific metrics to WandB every round
        self._log_wandb_metrics(infos)

        # convert outputs
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def _log_wandb_metrics(self, infos: List[Dict]) -> None:
        """Log per-episode bidding metrics directly to WandB."""
        try:
            import wandb
            if wandb.run is None:
                return
        except ImportError:
            return

        # Read ci_norm_range from config for Detector accuracy calculation
        ci_norm_range = 0.3
        try:
            ci_norm_range = self.config.env.bidding.reward.ci_norm_range
        except AttributeError:
            pass

        judge_a = [info.get("judge_score_a", 0.5) for info in infos]
        judge_b = [info.get("judge_score_b", 0.5) for info in infos]
        winner_rate = [float(info.get("won", 0.0)) for info in infos]

        metrics = {
            "bidding/winner_rate":       sum(winner_rate) / len(winner_rate),
            "bidding/judge_score_a_mean": sum(judge_a) / len(judge_a),
            "bidding/judge_score_b_mean": sum(judge_b) / len(judge_b),
        }

        # CI and Detector accuracy — only available on final round
        gaps = []
        ci_vals, cs_vals = [], []
        for info in infos:
            ci = info.get("ci")
            cs = info.get("collusion_score", 0.0)
            if ci is not None:
                ci_vals.append(ci)
                cs_vals.append(cs)
                normalized_ci = min(1.0, max(0.0, (ci - 1.0) / ci_norm_range))
                gaps.append(abs(cs - normalized_ci))

        if ci_vals:
            metrics["bidding/ci_mean"]             = sum(ci_vals) / len(ci_vals)
            metrics["bidding/collusion_score_mean"] = sum(cs_vals) / len(cs_vals)
            metrics["bidding/ci_detector_gap"]      = sum(gaps) / len(gaps)

        # Per-agent rewards
        rewards_a = [info["per_agent_rewards"]["BidderA"] for info in infos if "per_agent_rewards" in info]
        rewards_b = [info["per_agent_rewards"]["BidderB"] for info in infos if "per_agent_rewards" in info]
        rewards_d = [info["per_agent_rewards"]["Detector"] for info in infos if "per_agent_rewards" in info]
        if rewards_a:
            metrics["bidding/reward_bidder_a_mean"] = sum(rewards_a) / len(rewards_a)
            metrics["bidding/reward_bidder_b_mean"] = sum(rewards_b) / len(rewards_b)
            metrics["bidding/reward_detector_mean"] = sum(rewards_d) / len(rewards_d)

        # Bid values and economics
        bids_a  = [info.get("agent_A_bid") for info in infos]
        bids_b  = [info.get("agent_B_bid") for info in infos]
        costs   = [info.get("current_cost") for info in infos]
        gaps    = [info.get("bid_gap") for info in infos]

        valid_a = [b for b in bids_a if b is not None]
        valid_b = [b for b in bids_b if b is not None]
        if valid_a:
            metrics["bidding/bid_a_mean"] = sum(valid_a) / len(valid_a)
        if valid_b:
            metrics["bidding/bid_b_mean"] = sum(valid_b) / len(valid_b)

        # Bid as a multiple of cost (1.0 = no profit, 2.0 = 100% markup)
        ratio_a = [b / c for b, c in zip(bids_a, costs) if b is not None and c]
        ratio_b = [b / c for b, c in zip(bids_b, costs) if b is not None and c]
        if ratio_a:
            metrics["bidding/bid_a_over_cost_mean"] = sum(ratio_a) / len(ratio_a)
        if ratio_b:
            metrics["bidding/bid_b_over_cost_mean"] = sum(ratio_b) / len(ratio_b)

        # Bid gap — small gap means bids are converging (coordination signal)
        valid_gaps = [g for g in gaps if g is not None]
        if valid_gaps:
            metrics["bidding/bid_gap_mean"] = sum(valid_gaps) / len(valid_gaps)

        # Cost baseline
        valid_costs = [c for c in costs if c is not None]
        if valid_costs:
            metrics["bidding/cost_mean"] = sum(valid_costs) / len(valid_costs)

        # Win rates per bidder
        winners = [info.get("winner") for info in infos]
        n = len(winners)
        metrics["bidding/bidder_a_win_rate"] = sum(1 for w in winners if w == "BidderA") / n
        metrics["bidding/bidder_b_win_rate"] = sum(1 for w in winners if w == "BidderB") / n
        metrics["bidding/no_winner_rate"]    = sum(1 for w in winners if w is None) / n

        # Invalid bid rate — None bids mean format/projection failure
        total_slots = 2 * len(infos)
        n_invalid = sum(1 for b in bids_a if b is None) + sum(1 for b in bids_b if b is None)
        metrics["bidding/invalid_bid_rate"] = n_invalid / total_slots if total_slots else 0.0

        wandb.log(metrics, commit=False)
             

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        """
        This function builds the text observation (prompt) for the agent.
        
        Returns:
        - postprocess_text_obs (List[str]): A list of processed text observations.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length
            )

        for i in range(len(text_obs)):
            current_obs = text_obs[i] or ""

            # task only, no history
            if init or self.config.env.history_length <= 0:
                obs_i = BIDDING_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_obs=current_obs,
                )
            # agent sees history + step number
            else:
                obs_i = BIDDING_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]) + 1,
                    max_steps=getattr(self.config.env, "max_steps", 5),
                    current_obs=current_obs,
                )

            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        """
        Finds the last entry with active masks (evaluation)
        """
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]

            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]

                # used for metrics: success rate, evaluation stats (logging only)
                won_value = float(info.get("won", 0.0))
                success["success_rate"].append(won_value)

                data_source = info.get("data_source", "bidding")
                if data_source is not None:
                    success[f"{data_source}_success_rate"].append(won_value)
                
                return


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        if not self.config.agent.multi_agent:
            actions, valids = self.projection_f(text_actions)
        else:
            actions = text_actions

        time1 = time.time()
        next_obs, rewards, dones, infos = self.envs.step(actions)
        time2 = time.time()
        print(f"SearchEnv step time: {time2 - time1:.4f} seconds")

        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        if not self.config.agent.multi_agent:
            for i, info in enumerate(infos):
                info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                if self.config.agent.multi_agent:
                    obs_i = SEARCH_MULTIAGENT_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i]
                    )
                else:
                    obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i]
                    )
            else:
                if self.config.agent.multi_agent:
                    obs_i = SEARCH_MULTIAGENT_TEMPLATE.format(
                        task_description=self.tasks[i],
                        memory_context="{memory}" if self.config.agent.use_agent_memory else memory_ctx[i],
                        step_count=len(self.memory[i]),
                    )
                else:
                    obs_i = SEARCH_TEMPLATE.format(
                        task_description=self.tasks[i],
                        memory_context=memory_ctx[i],
                        step_count=len(self.memory[i]),
                    )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            


class MathEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for MathEnv.
    """
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        observations = {
            "text": self.build_text_obs(obs),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        if not self.config.agent.multi_agent:
            actions, valids = self.projection_f(text_actions)
        else:
            actions = text_actions

        time1 = time.time()
        next_obs, rewards, dones, infos = self.envs.step(actions)
        time2 = time.time()
        print(f"MathEnv step time: {time2 - time1:.4f} seconds")

        next_observations = {
            "text": None,
            "image": None,
            "anchor": None
        }
        
        if not self.config.agent.multi_agent:
            for i, info in enumerate(infos):
                info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        for i in range(len(text_obs)):
            if self.config.agent.multi_agent:
                obs_i = MATH_MULTIAGENT_TEMPLATE.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = MATH_TEMPLATE.format(
                    task_description=self.tasks[i]
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    # Get validation rollout n for pass@k and avg@k computation
    val_group_n = getattr(config.env.rollout, 'val_n', 1)

    if "math" in config.env.env_name.lower():
        from agent_system.environments.env_package.math import build_math_envs, math_projection
        _envs = build_math_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True)
        _val_envs = build_math_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=val_group_n, is_train=False)
        
        projection_f = partial(math_projection)
        envs = MathEnvironmentManager(_envs, projection_f, config)
        val_envs = MathEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=val_group_n, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    
    # ---------------------- OUR SETUP -----------------------------------------------------------------
    elif "bidding" in config.env.env_name.lower():
        from agent_system.environments.env_package.bidding import build_bidding_envs, bidding_projection
        _envs = build_bidding_envs(
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            is_train=True,
            env_config=config.env
        )
        _val_envs = build_bidding_envs(
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=val_group_n,
            is_train=False,
            env_config=config.env
        )
        projection_f = partial(bidding_projection)
        envs = BiddingEnvironmentManager(_envs, projection_f, config)
        val_envs = BiddingEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    # ---------------------- OUR SETUP ------------------------------------------------------------------

    else:
        print("Environment not supported")
        exit(1)