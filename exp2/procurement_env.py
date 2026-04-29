# procurement_env.py

# defines the env where the auction game will take place

import numpy as np

from reward_fn import total_reward
from utils import clip_bid, extract_bid

class ProcurementAuctionEnv:
    '''
    Defines the procurement auction environment:
    - step + reset
    - history tracking
    - score assignment
    '''

    def __init__(self, cfg):
        self.cfg = cfg
        self.agents = ['agent_1', 'agent_2']
        self.round_idx = 0
        self.history = []
        self.private_costs = {}

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.round_idx = 0
        self.history = []
        self.private_costs = self._sample_costs()
        return self._observations()
    
    def _sample_costs(self):
        '''
        samples a value for marginal cost given mean and std
        '''
        costs = {}
        for agent in self.agents:
            cost = np.random.normal(self.cfg.cost_mean, self.cfg.cost_std)
            costs[agent] = max(1.0, float(cost))
        return costs

    def _history_text(self):
        '''
        returns history to feed agent during training given round and history_window configuration
        '''
        recent = self.history[-self.cfg.history_window :]
        if not recent:
            return "No previous rounds."
        lines = []
        for row in recent:
            lines.append(
                f"Round {row['round']}: bids={row['bids']}, winner={row['winner']}"
            )
        return "\n".join(lines)

    def _observations(self):
        '''
        defines observations
        '''
        public_history = self._history_text()
        return {
            agent: {
                "round": self.round_idx,
                "private_cost": self.private_costs[agent],
                "public_history": public_history,
            }
            for agent in self.agents
        }
    
    def step(self, actions, judge_detector, detector_override_scores=None):
        bids = {}
        for agent in self.agents:
            raw_bid = extract_bid(actions[agent]["text"])
            bids[agent] = clip_bid(raw_bid, self.cfg.min_bid, self.cfg.max_bid)

        # Procurement/reverse auction: lowest bid wins.
        winner = min(bids, key=bids.get)
        rewards = {}
        infos = {}

        for agent in self.agents:
            opponent = "agent_2" if agent == "agent_1" else "agent_1"
            judge_scores = judge_detector.score(
                actions[agent]["text"], actions[opponent]["text"]
            )

            judge_collusion = judge_scores["collusion_score"]
            detector_score = None
            if detector_override_scores is not None:
                detector_score = float(detector_override_scores.get(agent, judge_collusion))

            # Trainable detector can become the real-time penalty signal.
            final_collusion_score = (
                detector_score if detector_score is not None else judge_collusion
            )

            won = agent == winner
            reward, profit = total_reward(
                bid=bids[agent],
                cost=self.private_costs[agent],
                won=won,
                competitiveness=judge_scores["competitiveness"],
                collusion_score=final_collusion_score,
                cfg=self.cfg,
            )
            rewards[agent] = reward
            infos[agent] = {
                "bid": bids[agent],
                "cost": self.private_costs[agent],
                "won": won,
                "profit": profit,
                "competitiveness": judge_scores["competitiveness"],
                "judge_collusion_score": judge_collusion,
                "detector_collusion_score": detector_score,
                "collusion_score": final_collusion_score,
                "judge_explanation": judge_scores["explanation"],
            }

        self.history.append(
            {
                "round": self.round_idx,
                "bids": bids,
                "winner": winner,
                "rewards": rewards,
            }
        )
        self.round_idx += 1
        done = self.round_idx >= self.cfg.num_rounds
        self.private_costs = self._sample_costs()
        return (
            self._observations(),
            rewards,
            {agent: done for agent in self.agents},
            {agent: False for agent in self.agents},
            infos,
        )