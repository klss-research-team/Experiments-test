# projection.py

from typing import List, Tuple
import re

def bidding_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """
    Project a list of LLM actions into (`results`, `valids`)
    """

    valids: List[int] = 1 * len(actions)

    return actions, valids

    # FROM OUTLINE FILE:
    # """
    # Trivial projection: all bidding responses are treated as valid.
    # In multi-agent mode DrMAS skips the projection, so this is only called
    # in single-agent evaluation mode.
    # """
    # valids = np.ones(len(text_responses), dtype=bool)
    # return text_responses, valids