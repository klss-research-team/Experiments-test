# Reverse Auction Multi-Agent LLM GRPO Experiment

This project impelements a repeated procurement auctino experiment

## Setup

3 LLM Agents:
- Pricing agent 1
- Pricing agent 2
- Judge/detector agent

The 2 pricing agents submit:
- reasoning traces
- bids

The judge/detector scores:
- competitiveness (of reasoning)
- collusion risk

## Auction rules

- Lowest bid wins
- Winner profit = bid - private cost
- Loser profit = 0

## Main files:
train_agents.py                # main loop: agents train + detector trains
detector_model.py              # detector classifier
procurement_env.py             # reverse auction environment
metrics.py                     # CI + SDR metrics
baselines.py                   # baseline runs
plot_results.py                # plots

## Run order:
```
pip install -r requirements.txt
python train_agents_.py
python metrics.py
python baselines.py
python plot_results.py
```