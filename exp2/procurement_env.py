# procurement_env.py

# defines the env where the auction game will take place

import numpy as np

from reward_fn import total_reward
from utils import clip_bid, extract_bid

class ProcurementAuctionEnv:
    '''
    Defines the procurement auction environment:
    - step + reset
    - history tracking
    - score assignment
    '''