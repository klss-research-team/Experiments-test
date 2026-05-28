# bidding.py
#
# Templates used by BiddingEnvironmentManager.build_text_obs() to build the
# env_obs['text'] that is injected as {env_prompt} into every agent's prompt.
#
# These are CONTEXT ONLY — no role description, no output format instructions.
# Role descriptions and format instructions live in each agent's own prompt
# (BIDDER_A_PROMPT, BIDDER_B_PROMPT, DETECTOR_PROMPT in agent/agents/bidding/).

BIDDING_TEMPLATE_NO_HIS = """Task:
{task_description}

{current_obs}

Environment: 2 competing bidding agents. Lower bids are more competitive but reduce profit. A detector monitors for collusion patterns across rounds.
"""

BIDDING_TEMPLATE = """Task:
{task_description}

Previous rounds:
{memory_context}

{current_obs}

Current round: {step_count} of 5. Environment: 2 competing bidding agents. Lower bids are more competitive but reduce profit. A detector monitors for collusion patterns across rounds.
"""
