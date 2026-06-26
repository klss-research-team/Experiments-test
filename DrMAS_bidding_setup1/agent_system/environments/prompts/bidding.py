# bidding.py
#
# Templates used by BiddingEnvironmentManager.build_text_obs() to build the
# env_obs['text'] that is injected as {env_prompt} into every agent's prompt.
#
# These are CONTEXT ONLY — no role description, no output format instructions.
# Role descriptions and format instructions live in each agent's own prompt
# (BIDDER_A_PROMPT, BIDDER_B_PROMPT, DETECTOR_PROMPT in agent/agents/bidding/).

BIDDING_TEMPLATE_NO_HIS = """Contract type: {task_description}

=== BID NOW ===
{current_obs}
"""

BIDDING_TEMPLATE = """Contract type: {task_description}

=== PREVIOUS ROUNDS (for context only — do NOT use these costs or budgets) ===
{memory_context}

=== BID NOW — ROUND {step_count} of {max_steps} ===
{current_obs}
"""
