# utils.py

"""
Helper functions for calculating agent rewards
"""

import re

# TODO:
def parse_bid(text: str) -> str:
    pass

# TODO:
def parse_response(text: str) -> str;
    pass

def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False