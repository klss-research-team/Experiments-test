# projection.py

import re
import numpy as np
from typing import List, Tuple, Dict, Any


def _has_valid_think(text: str) -> bool:
    """Check for exactly one valid <think>...</think> block."""
    think_start_count = text.count("<think>")
    think_end_count = text.count("</think>")
    think_start_idx = text.find("<think>")
    think_end_idx = text.find("</think>")

    return (
        think_start_count == 1
        and think_end_count == 1
        and think_start_idx != -1
        and think_end_idx != -1
        and think_end_idx > think_start_idx
    )


def _extract_tag(text: str, tag: str) -> str:
    """Extract content from the first <tag>...</tag> block."""
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if match is None:
        return ""

    return match.group(1).strip()


def _parse_float(value: str):
    """Parse a numeric string like '$1,200.50' into float."""
    if value is None or value == "":
        return None

    cleaned = value.strip().replace("$", "").replace(",", "")

    try:
        return float(cleaned)
    except ValueError:
        return None


def bidding_projection(
    actions: List[str],
    check_think_tag: bool = False,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Project bidder LLM responses into structured bidding actions.

    Expected format:
    <think>...</think>
    <bid>numeric bid</bid>
    <reasoning>public reasoning</reasoning>

    Returns
    -------
    projected_actions : List[Dict[str, Any]]
        Each dict contains bid, reasoning, and raw_response.
    valids : np.ndarray
        Boolean array indicating whether each response was valid.
    """
    projected_actions = []
    valids = []

    for original_str in actions:
        is_valid = True

        if check_think_tag and not _has_valid_think(original_str):
            is_valid = False

        bid_text = _extract_tag(original_str, "bid")
        reasoning = _extract_tag(original_str, "reasoning")

        bid = _parse_float(bid_text)

        if bid is None:
            is_valid = False
        elif bid <= 0:
            is_valid = False
            bid = None

        projected_actions.append({
            "bid": bid,
            "reasoning": reasoning,
            "raw_response": original_str,
        })

        valids.append(is_valid)

    return projected_actions, np.array(valids, dtype=bool)


def detector_projection(
    actions: List[str],
    check_think_tag: bool = False,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Project detector LLM responses into structured detector outputs.

    Expected format:
    <think>...</think>
    <collusion_score>0 to 1</collusion_score>
    <explanation>short explanation</explanation>

    Returns
    -------
    projected_actions : List[Dict[str, Any]]
        Each dict contains collusion_score, explanation, and raw_response.
    valids : np.ndarray
        Boolean array indicating whether each response was valid.
    """
    projected_actions = []
    valids = []

    for original_str in actions:
        is_valid = True

        if check_think_tag and not _has_valid_think(original_str):
            is_valid = False

        score_text = _extract_tag(original_str, "collusion_score")
        explanation = _extract_tag(original_str, "explanation")

        collusion_score = _parse_float(score_text)

        if collusion_score is None:
            is_valid = False
            collusion_score = 0.0
        else:
            # clamp to [0, 1]
            collusion_score = max(0.0, min(1.0, collusion_score))

        projected_actions.append({
            "collusion_score": collusion_score,
            "explanation": explanation,
            "raw_response": original_str,
        })

        valids.append(is_valid)

    return projected_actions, np.array(valids, dtype=bool)