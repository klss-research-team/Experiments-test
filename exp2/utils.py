# utils.py

# defines utility functions to use throughout experiment

import json
import os
import re

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def extract_bid(text):
    match = re.search(r"Bid:\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    if match is None:
        match = re.search(r"\$?\s*([0-9]+(?:\.[0-9]+)?)", text)
    if match is None:
        return None
    return float(match.group(1))

def clip_bid(bid, min_bid, max_bid):
    if bid is None:
        return max_bid
    return max(min_bid, min(max_bid, bid))

def safe_json_loads(text, default):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return default
    
def group_normalize(values):
    mean = values.mean()
    std = values.std(unbiased=False)
    if std.item() < 1e-8:
        return values * 0.0
    return (values - mean) / (std + 1e-8)