# projection.py

import re
import numpy as np
from typing import List, Tuple, Dict, Any


def _extract_tag(text: str, tag: str) -> str:
    """Extract content from the first <tag>...</tag> block (XML fallback)."""
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_bold_field(text: str, field: str) -> str:
    """
    Extract content after a **Field:** markdown bold header (primary format).
    Uses [*:\s]+ after the field name to consume any mix of *, :, and spaces,
    which handles both **Field:** (colon inside bold) and **Field**: variants.
    Captures until the next **Bold:** section, ### header, or end of string.
    """
    escaped = re.escape(field)
    pattern = rf'\*\*{escaped}[*:\s]+(.*?)(?=\n\s*\*\*[A-Za-z]|\n\s*###|\Z)'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_section_header(text: str, field: str) -> str:
    """Extract content after a ### Field: markdown section header (secondary fallback)."""
    escaped = re.escape(field)
    pattern = rf'###\s*{escaped}\s*:?\s*\n?\s*(.*?)(?=\n\s*###|\n\s*\*\*|\Z)'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_float(value: str):
    """Parse a numeric string like '$1,200.50' into float."""
    if not value:
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

    Primary format (markdown bold, matches 1.5B model output):
    **Think:** strategic reasoning (max 80 words)
    **Bid:** numeric value only
    **Reasoning:** one short public justification (max 20 words)

    XML tags are accepted as a fallback for larger models.

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

        # Bid: try markdown bold first, then XML tag
        bid_text = _extract_bold_field(original_str, "Bid") or _extract_tag(original_str, "bid")
        bid = _parse_float(bid_text)

        # Reasoning: try markdown bold first, then XML tag
        reasoning = _extract_bold_field(original_str, "Reasoning") or _extract_tag(original_str, "reasoning")

        if bid is None or bid <= 0:
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

    Primary format (markdown bold, matches 1.5B model output):
    **Think:** evidence-based analysis
    **Collusion Score:** 0.0 to 1.0
    **Explanation:** 1-2 sentence explanation

    XML tags and ### section headers are accepted as fallbacks.

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

        # Collusion score: markdown bold → ### section header → XML tag
        score_text = (
            _extract_bold_field(original_str, "Collusion Score")
            or _extract_section_header(original_str, "Collusion Score")
            or _extract_tag(original_str, "collusion_score")
        )
        explanation = (
            _extract_bold_field(original_str, "Explanation")
            or _extract_section_header(original_str, "Explanation")
            or _extract_tag(original_str, "explanation")
        )

        collusion_score = _parse_float(score_text)

        if collusion_score is None:
            is_valid = False
            collusion_score = 0.0
        else:
            collusion_score = max(0.0, min(1.0, collusion_score))

        projected_actions.append({
            "collusion_score": collusion_score,
            "explanation": explanation,
            "raw_response": original_str,
        })

        valids.append(is_valid)

    return projected_actions, np.array(valids, dtype=bool)
