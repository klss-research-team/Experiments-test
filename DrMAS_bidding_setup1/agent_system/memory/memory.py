# Copyright 2025 Nanyang Technological University (NTU), Singapore
# Copyright 2025 verl-agent (GiGPO) Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Dict, Any, Tuple
from .base import BaseMemory

class BiddingMemory(BaseMemory):
    """
    Memory manager: responsible for storing and fetching per-environment history records
    """
    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        """
        Return the number of memory slots.
        """
        return 0 if self._data is None else len(self._data)

    def __getitem__(self, idx: int):
        """
        Access memory of specific environment index.
        """
        return self._data[idx]

    def reset(self, batch_size: int):
        """
        Reset memory with given batch size.
        """
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Stores a new record (one step of history) for each environment instance
        """
        if self.keys is None:
            self.keys = list(record.keys())

        assert self.keys == list(record.keys()), (
            f"Record keys {list(record.keys())} do not match expected keys {self.keys}"
        )

        for key, values in record.items():
            assert len(values) == self.batch_size, (
                f"Key '{key}' has length {len(values)}, expected {self.batch_size}"
            )

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({
                k: record[k][env_idx] for k in self.keys
            })

    def fetch(self, history_length: int) -> Tuple[List[str], List[int]]:
        """
        Fetches recent auction history for each environment.

        Parameters
        ----------
        record : int
            Maximum numner of past rounds to include.

        Returns
        -------
        memory_contexts: List[str]
            A list of formatted action history strings (one for each environment). 
        valid_lengths : List[int]
            A list of the actual number of valid rounds actually included (one for each environment).
        """
        memory_contexts = []
        valid_lengths = []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []

            for j, rec in enumerate(recent):
                round_num = start_idx + j + 1

                lines.append(
                f"Round {round_num}:\n"
                f"Agent A bid: {rec.get('agent_A_bid')}\n"
                f"Agent A reasoning: {rec.get('agent_A_reasoning')}\n"
                f"Agent B bid: {rec.get('agent_B_bid')}\n"
                f"Agent B reasoning: {rec.get('agent_B_reasoning')}\n"
                f"Winner: {rec.get('winner')}\n"
                f"Detector score: {rec.get('collusion_score')}\n"
                )
            
            memory_contexts.append("\n".join(lines)) # converts history -> str for LLM inputs
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths
    
class SimpleMemory(BaseMemory):
    """
    Memory manager: responsible for storing & fetching per‑environment history records.
    """
    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'text_obs', 'action'), and each value is a list of 
                length `batch_size`, containing the data for each environment.
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str = "text_obs",
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch and format recent interaction history for each environment instance.
        Args:
            history_length (int):
                Maximum number of past steps to retrieve per environment.
            obs_key (str, default="text_obs"):
                The key name used to access the observation in stored records.
                For example: "text_obs" or "Observation", depending on the environment.
            action_key (str, default="action"):
                The key name used to access the action in stored records.
                For example: "action" or "Action".
        Returns:
            memory_contexts : List[str]
                A list of formatted action history strings for each environment.
            valid_lengths : List[int]
                A list of the actual number of valid history steps per environment.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                lines.append(
                    f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"
                )

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths
    

class SearchMemory(BaseMemory):
    """
    Memory manager: responsible for storing & fetching per‑environment history records.
    """
    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'text_obs', 'action'), and each value is a list of 
                length `batch_size`, containing the data for each environment.
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str,
        action_key: str,
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch and format recent interaction history for each environment instance.
        Args:
            history_length (int):
                Maximum number of past steps to retrieve per environment.
            obs_key (str):
                The key name used to access the observation in stored records.
                For example: "text_obs" or "Observation", depending on the environment.
            action_key (str):
                The key name used to access the action in stored records.
                For example: "action" or "Action".
        Returns:
            memory_contexts : List[str]
                A list of formatted action history strings for each environment.
            valid_lengths : List[int]
                A list of the actual number of valid history steps per environment.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                lines.append(
                    f"Step {step_num}:{act} {obs}\n"
                    # f"{act}\n{obs}\n"
                )

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths