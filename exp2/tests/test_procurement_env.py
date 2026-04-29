import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs import AuctionConfig
from procurement_env import ProcurementAuctionEnv

class DummyJudgeDetector:
    def score(self, text_a, text_b):
        return {
            "collusion_score": 0.1,
            "competitiveness": 0.9,
            "explanation": "dummy judge score",
        }


def make_config():
    config = AuctionConfig()
    config.cost_mean = 10.0
    config.cost_std = 1.0
    config.num_rounds = 2
    
    return config


def test_reset_returns_observations():
    config = make_config()
    env = ProcurementAuctionEnv(config)

    obs = env.reset(seed=42)

    assert "agent_1" in obs
    assert "agent_2" in obs
    assert obs["agent_1"]["round"] == 0
    assert "private_cost" in obs["agent_1"]
    assert obs["agent_1"]["public_history"] == "No previous rounds."


def test_step_lowest_bid_wins():
    config = make_config()
    env = ProcurementAuctionEnv(config)
    env.reset(seed=42)

    actions = {
        "agent_1": {"text": "Bid: 20, Reasoning: trying to win."},
        "agent_2": {"text": "Bid: 30, Reasoning: higher margin."},
    }

    obs, rewards, terminated, truncated, infos = env.step(
        actions,
        judge_detector=DummyJudgeDetector(),
    )

    assert infos["agent_1"]["bid"] == 20
    assert infos["agent_2"]["bid"] == 30
    assert infos["agent_1"]["won"] is True
    assert infos["agent_2"]["won"] is False

    assert len(env.history) == 1
    assert env.history[0]["winner"] == "agent_1"

    assert terminated["agent_1"] is False
    assert truncated["agent_1"] is False


def test_done_after_num_rounds():
    config = make_config()
    env = ProcurementAuctionEnv(config)
    env.reset(seed=42)

    actions = {
        "agent_1": {"text": "Bid: 20"},
        "agent_2": {"text": "Bid: 30"},
    }

    env.step(actions, DummyJudgeDetector())
    obs, rewards, terminated, truncated, infos = env.step(
        actions,
        DummyJudgeDetector(),
    )

    assert terminated["agent_1"] is True
    assert terminated["agent_2"] is True