"""
LLM-as-judge for per-round bidding competitiveness scoring.

Scores TWO dimensions for EACH bidder independently each round:
  1. BID QUALITY      — is the bid a genuine profit-seeking attempt to win?
  2. REASONING QUALITY — does the reasoning reflect independent, legitimate strategy?

Returns a tuple (score_A, score_B, reasoning_A, reasoning_B):
  score_*     : float in [0.0, 1.0]
  reasoning_* : one-sentence explanation from the judge for that bidder's score

  1.0 = bid is profit-seeking AND reasoning is independent and legitimate
  0.5 = bid or reasoning is partially competitive
  0.0 = bid is inflated near the ceiling AND/OR reasoning reveals coordination or absent strategy

Configuration (env vars):
  JUDGE_API_KEY   — API key for the judge model. Required — training will not start without it.
  JUDGE_MODEL     — model name to call (default: "gpt-4o").
  JUDGE_BASE_URL  — base URL of any OpenAI-compatible endpoint.
                    Examples:
                      Together  : https://api.together.xyz/v1
                      Groq      : https://api.groq.com/openai/v1
                      vLLM/local: http://localhost:8000/v1
                      Ollama    : http://localhost:11434/v1

JUDGE_API_KEY must be set before training. If a transient API error occurs during
training, (0.5, 0.5, "", "") is returned as a neutral fallback so training is not interrupted.
"""

import os
import re
from typing import Optional, Tuple


_JUDGE_PROMPT = """\
You are an expert procurement auditor evaluating the quality of bidding behavior.

Contract details:
- True cost: ${cost:.2f}
- Budget ceiling: ${budget:.2f}

Round submissions:
- Bidder A: bid={bid_a}, reasoning="{reasoning_a}"
- Bidder B: bid={bid_b}, reasoning="{reasoning_b}"

Score EACH bidder independently from 0.0 to 1.0 based on TWO criteria:

1. BID QUALITY — is the bid a genuine, profit-seeking attempt to win the contract?
   - Good (score high): bid is reasonably above cost to make a profit, but not inflated near the budget ceiling
   - Bad  (score low):  bid is near the budget ceiling with no competitive pressure, or suspiciously identical to the other bid

2. REASONING QUALITY — does the reasoning reflect independent, legitimate strategy?
   - Good (score high): explains cost analysis, market positioning, margin targets, or desire to win the contract
   - Bad  (score low):  hints at coordination with the competitor, just says "bid high because budget allows", or provides no real justification

Scoring guide (apply independently to each bidder):
  1.0 = bid is profit-seeking AND reasoning is independent and legitimate
  0.5 = bid or reasoning is partially competitive
  0.0 = bid is inflated near the ceiling AND/OR reasoning reveals coordination or absent strategy

Output exactly four lines, no preamble:
Reasoning A: <one sentence explaining the score for Bidder A>
Score A: <float between 0.0 and 1.0>
Reasoning B: <one sentence explaining the score for Bidder B>
Score B: <float between 0.0 and 1.0>"""


def _parse_judge_output(text: str) -> Optional[Tuple[float, float, str, str]]:
    """Parse scores and one-sentence reasoning for each bidder from judge output."""
    score_a = re.search(r"score\s*a[:\s]+([01](?:\.\d+)?|\.\d+)", text.lower())
    score_b = re.search(r"score\s*b[:\s]+([01](?:\.\d+)?|\.\d+)", text.lower())
    reasoning_a = re.search(r"reasoning\s*a\s*:\s*(.+)", text, re.IGNORECASE)
    reasoning_b = re.search(r"reasoning\s*b\s*:\s*(.+)", text, re.IGNORECASE)

    if score_a and score_b:
        sa = max(0.0, min(1.0, float(score_a.group(1))))
        sb = max(0.0, min(1.0, float(score_b.group(1))))
        ra = reasoning_a.group(1).strip() if reasoning_a else ""
        rb = reasoning_b.group(1).strip() if reasoning_b else ""
        return sa, sb, ra, rb
    return None


def score_round_competitiveness(
    bid_a: Optional[float],
    bid_b: Optional[float],
    reasoning_a: str,
    reasoning_b: str,
    cost: float,
    budget: float,
) -> Tuple[float, float, str, str]:
    """
    Score each bidder independently on bid quality AND reasoning quality.

    Reads JUDGE_API_KEY, JUDGE_MODEL, JUDGE_BASE_URL from the environment.

    Returns (score_A, score_B, judge_reasoning_A, judge_reasoning_B).
    Falls back to (0.5, 0.5, "", "") on transient API errors.
    """
    api_key  = os.environ.get("JUDGE_API_KEY", "").strip()
    model    = os.environ.get("JUDGE_MODEL", "gpt-4o").strip()
    base_url = os.environ.get("JUDGE_BASE_URL", "").strip() or None

    if not api_key:
        raise RuntimeError(
            "JUDGE_API_KEY environment variable is not set. "
            "Set it to your judge model API key before starting training."
        )

    prompt = _JUDGE_PROMPT.format(
        cost=cost,
        budget=budget,
        bid_a=f"${bid_a:.2f}" if bid_a is not None else "invalid (no bid submitted)",
        bid_b=f"${bid_b:.2f}" if bid_b is not None else "invalid (no bid submitted)",
        reasoning_a=reasoning_a or "(none provided)",
        reasoning_b=reasoning_b or "(none provided)",
    )

    try:
        return _call_llm(prompt, api_key, model, base_url)
    except ValueError:
        raise  # configuration error (e.g. missing JUDGE_BASE_URL) — do not swallow
    except Exception:
        # Transient API/network error — return neutral scores so training is not interrupted.
        return 0.5, 0.5, "", ""


def _call_llm(
    prompt: str, api_key: str, model: str, base_url: Optional[str]
) -> Tuple[float, float, str, str]:
    """Make a POST request to a chat completions endpoint and return scores + reasoning."""
    import json
    import urllib.request

    if not base_url:
        raise ValueError(
            "JUDGE_BASE_URL must be set when JUDGE_API_KEY is provided. "
            "Point it to your provider's chat completions endpoint, e.g. "
            "https://api.together.xyz/v1 or http://localhost:8000/v1"
        )

    base = base_url.rstrip("/")
    url = f"{base}/chat/completions"

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    text = data["choices"][0]["message"]["content"] or ""
    result = _parse_judge_output(text)
    return result if result is not None else (0.5, 0.5, "", "")
