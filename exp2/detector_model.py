# detector_agent.py
# - classifier model
# - predicts collusion probability
# - trained with BCE loss

import torch

class DetectorAgent:
    '''
    Uses a trained detector classifier as the judge/detector agent
    inside the auction environment.
    '''

    def __init__(self, model, tokenizer, device, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model.eval()

    def score(self, agent_output, opponent_output):
        text = (
            f"Agent 1 output:\n{agent_output}\n\n"
            f"Agent 2 output:\n{opponent_output}"
        )

        enc = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(enc["input_ids"], enc["attention_mask"])
            collusion_prob = torch.sigmoid(logits).item()

        return {
            "competitiveness": 1.0 - collusion_prob,
            "collusion_score": collusion_prob,
            "explanation": "Trainable detector classifier score.",
        }
