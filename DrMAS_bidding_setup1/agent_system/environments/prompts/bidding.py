# bidding.py
#
# Templates used by BiddingEnvironmentManager.build_text_obs() to build the
# env_obs['text'] that is injected as {env_prompt} into every agent's prompt.
#
# These are CONTEXT ONLY — no role description, no output format instructions.
# Role descriptions and format instructions live in each agent's own prompt
# (BIDDER_A_PROMPT, BIDDER_B_PROMPT, DETECTOR_PROMPT in agent/agents/bidding/).

BIDDING_TEMPLATE_NO_HIS = """Contract type: {task_description}

=== CURRENT ROUND ===
{current_obs}
"""

BIDDING_TEMPLATE = """Contract type: {task_description}

=== PREVIOUS ROUNDS (history only — costs and budgets below are from past rounds, NOT the current one) ===
{memory_context}

=== CURRENT ROUND {step_count} of {max_steps} ===
{current_obs}
"""
