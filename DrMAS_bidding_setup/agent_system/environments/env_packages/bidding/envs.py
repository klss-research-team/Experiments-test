# envs.py

import asyncio
import concurrent.futures
from typing import Any, Dict, List
import numpy as np
# from agent_system.environments.env_package.math.math_reward import RewardConfig, RewardMathFn
from agent_system.environments.env_package.bidding.bidding_reward import RewardConfig, RewardBiddingFn

# TODO: adapt MathEnv and MathMultiProcessEnv -> BiddingEnv and BiddingMultiProcessEnv
# go over the entire BiddingEnv (help)
class BiddingEnv:
    """
    Environment for bidding tasks.

    Holds the state for one contract episode.

    The orchestra handles all multi-turn bidding interanlly, so this environment only needs to:
        - reset(): supply the contract as the initial observation
        - step(): signal done=True (rewards written by the orchestra)
    """
    def __init__(self,):
        self.ground_truth = None # ?? profit?
        self.contract = Dict[str, Any] = {} # added from outline
        self.data.source = "bidding"
        reward_config = RewardConfig()
        self.reward_fn = RewardBiddingFn(reward_config)

    def reset(self, extras: Dict[str, Any]) -> str:
        self.contract = extras["contract"]
        self.data_source = extras.get("data_source", "bidding")
        obs = self._format_contract(self.contract)
        return obs, {"data source": self.data_source}

        self.ground_truth = extras["ground_truth"] # ??? kept from math, idk

    def step(self, action: str):
        done = True # always done after one step
        task_info = {
            "ground_truth": self.ground_truth,
            "has_toolcall": False # TODO: change this for bidding env
        }

        profit_reward, win_reward, competivieness_reward, detection_penalty = self.reward_fn(task_info, action) # TODO: define reward_fn()
        
        # TODO: compute total reward here
        # total_reward = profit_weight * profit_reward + win_weight * win_reward + competitiveness_weight * competitiveness_reward - (detection_weight + detection_reward)

        obs = None
        info = {"data_source": self.data_source, "won": win_reward} # add more from reward function; change according to total_reward
        return obs, total_reward, done, info # update later
    
    # TODO
    # def _format_contract(self, contract: Dict) -> str:
    #     return (
    #         f"Contract ID": 
    #     )
    
    # TODO: fill in
    def close(self) -> None:
        pass


class BiddingMultiProcessEnv:
    """
    A simple mutli-environment wrapper running BiddingEnv in parallel threads.

    - env_num: logical number of groups (kept for external compatibility)
    - group_n: envs per group
    - total_envs: env_num * group_n
    """
    pass

    def __init__(
            self,
            seed: int = 0,
            env_num: int = 1,
            group_n: int = 1,
            is_train: bool = True
    ) -> None:
        super().__init__()

        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train

        self.envs = [BiddingEnv() for _ in range(self.batch_size)]

        max_workers = min(self.batch_size, 256)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    # TODO: fill in
    def _sync_reset(self, env, kwards):
        pass

    # TODO: fill in
    def _sync_step(self, env, action, kwargs):
        return env.step(action)

    # TODO: fill in
    def reset(self, kwargs: List[Dict]):
        pass
        
    # TODO: fill in
    def step(self, actions: List[str]):
        pass

    def close(self):
        if getattr(self, "_closed", False):
            return
        for env in self.envs:
            env.close()
        self._executor.shutdown(wait=True)
        self._loop.close()
        self._closed = True

    def __del__(self):
        self.close()


def build_bidding_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_config=None
    ) -> BiddingMultiProcessEnv:

    return BiddingMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
    )


# ------------- MATH ENV BELOW -------------------


class MathMultiProcessEnv:
    """
    A simple multi-environment wrapper running MathEnv in parallel threads.

    - env_num: logical number of groups (kept for external compatibility)
    - group_n: envs per group
    - total_envs = env_num * group_n
    """

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
    ) -> None:
        super().__init__()

        self.env_num   = env_num
        self.group_n   = group_n
        self.batch_size = env_num * group_n
        self.is_train  = is_train

        self.envs = [MathEnv() for _ in range(self.batch_size)]

        max_workers = min(self.batch_size, 256)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def _sync_reset(self, env, kwargs):
        extras = {
            "ground_truth": kwargs["ground_truth"],
            "data_source": kwargs.get("data_source", "unknown")
        }
        env.reset(extras)
        obs = kwargs["question"]
        info = {'data_source': kwargs.get("data_source", "unknown")}
        return obs, info
    
    def _sync_step(self, env, action: str):
        obs, reward, done, info = env.step(action)
        return obs, reward, done, info

    def reset(self, kwargs: List[Dict]):
        if len(kwargs) > self.batch_size:
            raise ValueError(f"Got {len(kwargs)} kwarg dicts, but the env was initialised with total_envs={self.batch_size}")

        pad_n = self.batch_size - len(kwargs)
        dummy_kw = {
                    "ground_truth": "",
                    "question": "",
                    "data_source": "unkown",
                }


        padded_kwargs = list(kwargs) + [dummy_kw] * pad_n
        valid_mask = [True] * len(kwargs) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_reset, env, kw)
            for env, kw in zip(self.envs, padded_kwargs)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, info_list = map(list, zip(*results))

        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]

        return obs_list, info_list

    def step(self, actions: List[str]):
        if len(actions) > self.batch_size:
            raise ValueError(f"Got {len(actions)} actions, but the env was initialized with total_envs={self.batch_size}")

        pad_n = self.batch_size - len(actions)
        padded_actions = list(actions) + [""] * pad_n
        valid_mask = [True] * len(actions) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_step, env, act)
            for env, act in zip(self.envs, padded_actions)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, reward_list, done_list, info_list = map(list, zip(*results))

        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        reward_list = [r for r, keep in zip(reward_list, valid_mask) if keep]
        done_list = [d for d, keep in zip(done_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]

        return obs_list, reward_list, done_list, info_list

    def close(self):
        if getattr(self, "_closed", False):
            return
        for env in self.envs:
            env.close()
        self._executor.shutdown(wait=True)
        self._loop.close()
        self._closed = True

    def __del__(self):
        self.close()


def build_math_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
):
    return MathMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
    )


class MathEnv:
    """
    Environment for Math execution tasks.
    """

    def __init__(self,):
        self.ground_truth = None
        self.data_source = "unknown"
        reward_config = RewardConfig()
        self.reward_fn = RewardMathFn(reward_config)


    def reset(self, extras: Dict[str, Any]) -> str:
        self.ground_truth = extras["ground_truth"]
        self.data_source = extras.get("data_source", "unknown")

    def step(self, action: str):
        done = True  # always done after one step
        task_info = {
            "ground_truth": self.ground_truth,
            "has_toolcall": False,
        }
        reward, is_correct = self.reward_fn(task_info, action)
        obs = None
        info = {"data_source": self.data_source, "won": is_correct}
        return obs, reward, done, info

    def close(self) -> None:
        pass