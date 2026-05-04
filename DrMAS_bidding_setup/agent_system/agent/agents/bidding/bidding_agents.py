from typing import Dict, Any, List, Tuple
from verl import DataProto
from transformers import PreTrainedTokenizer
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.agents.base import BaseAgent
from agent_system.agent.utils import math_projection
import numpy as np

BIDDER_PROMPT="""

(insert bidder agent prompt here)

"""

