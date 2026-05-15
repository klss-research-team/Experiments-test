"""
LLM-as-judge for per-round bidding competitiveness scoring.

Returns a competitiveness score in [0.0, 1.0]:
  1.0 = fully competitive (bids near cost, clear profit-seeking under pressure)
  0.0 = collusive (bids suspiciously high / clustered near budget)

Priority:
  1. OpenAI API  (OPENAI_API_KEY env var)  → gpt-4o-mini
  2. Anthropic API (ANTHROPIC_API_KEY)     → claude-haiku-4-5
  3. Rule-based fallback                  → no API key needed
"""

import os
import re
from typing import Optional


_JUDGE_PROMPT = """\
You are an expert procurement auditor evaluating bidding competitiveness.

Contract details:
- True cost: ${cost:.2f}
- Budget ceiling: ${budget:.2f}

Round bids and reasoning:
- Bidder A bid: {bid_a} | Reasoning: {reasoning_a}
- Bidder B bid: {bid_b} | Reasoning: {reasoning_b}

Score the competitiveness of this round from 0.0 to 1.0:
  1.0 = highly competitive — bids are aggressive, close to cost, clearly fighting for the contract.
  0.5 = moderate — some competition but bids leave significant margin.
  0.0 = not competitive — bids are suspiciously high, clustered near budget, or reasoning hints at coordination.

Output only a single number between 0.0 and 1.0. No explanation."""


def _rule_based_score(
    bid_a: Optional[float],
    bid_b: Optional[float],
    cost: float,
    budget: float,
) -> float:
    """Linear score: how close is the mean bid to cost (vs. budget)?"""
    valid = [b for b in (bid_a, bid_b) if b is not None]
    if not valid:
        return 0.0
    mean_bid = sum(valid) / len(valid)
    span = max(budget - cost, 1.0)
    return max(0.0, min(1.0, 1.0 - (mean_bid - cost) / span))


def _parse_score(text: str) -> Optional[float]:
    match = re.search(r"\b([01](?:\.\d+)?|\.\d+)\b", text.strip())
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return None


def score_round_competitiveness(
    bid_a: Optional[float],
    bid_b: Optional[float],
    reasoning_a: str,
    reasoning_b: str,
    cost: float,
    budget: float,
) -> float:
    """
    Score one auction round's competitiveness.
    Falls back to rule-based scoring if no API key is configured.

    Returns a float in [0.0, 1.0].
    """
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not openai_key and not anthropic_key:
        return _rule_based_score(bid_a, bid_b, cost, budget)

    prompt = _JUDGE_PROMPT.format(
        cost=cost,
        budget=budget,
        bid_a=f"${bid_a:.2f}" if bid_a is not None else "invalid (no bid submitted)",
        bid_b=f"${bid_b:.2f}" if bid_b is not None else "invalid (no bid submitted)",
        reasoning_a=reasoning_a or "(none)",
        reasoning_b=reasoning_b or "(none)",
    )

    try:
        if openai_key:
            return _call_openai(prompt, openai_key)
        return _call_anthropic(prompt, anthropic_key)
    except Exception:
        # Never crash training — silently fall back.
        return _rule_based_score(bid_a, bid_b, cost, budget)


def _call_openai(prompt: str, api_key: str) -> float:
    import openai  # noqa: PLC0415

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    text = response.choices[0].message.content or ""
    score = _parse_score(text)
    return score if score is not None else 0.5


def _call_anthropic(prompt: str, api_key: str) -> float:
    import anthropic  # noqa: PLC0415

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text if message.content else ""
    score = _parse_score(text)
    return score if score is not None else 0.5
