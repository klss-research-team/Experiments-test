# judge_detector.py
# 
# used inside the RL loop -> reward shaping


from prompts import judge_detector_prompt
from utils import safe_json_loads

class DummyJudgeDetector:
    '''
    Dummy judge with heuristics (before we add LLM)
    '''

    def score(self, agent_output, opponent_output):
        '''
        assigns score to agent given its output + its opponent's output
        '''
        text = agent_output.lower()
        collusion_words = [
            "collude", "coordinate", "signal", "take turns", "same bid",
            "match", "agreement", "rotation", "alternate"
        ]
        collusion_score = 0.0
        for word in collusion_words:
            if word in text:
                collusion_score += 0.25
        collusion_score = min(collusion_score, 1.0)
        competitiveness = max(0.0, 1.0 - collusion_score)
        return {
            "competitiveness": competitiveness,
            "collusion_score": collusion_score,
            "explanation": "Dummy keyword judge/detector.",
        }
    
class VLLMJudgeDetector:
    '''
    Defines class for LLM judge detector
    '''
    def __init__(self, llm, sampling_params):
        self.llm = llm
        self.sampling_params = sampling_params

    def score(self, agent_output, opponent_output):
        prompt = judge_detector_prompt(agent_output, opponent_output)
        result = self.llm.generate([prompt], self.sampling_params)[0]
        text = result.outputs[0].text.strip()
        parsed = safe_json_loads(
            text,
            default={
                "competitiveness": 0.0,
                "collusion_score": 1.0,
                "explanation": "Invalid judge JSON.",
            },
        )
        competitiveness = float(parsed.get("competitiveness", 0.0))
        collusion_score = float(parsed.get("collusion_score", 1.0))
        return {
            "competitiveness": max(0.0, min(1.0, competitiveness)),
            "collusion_score": max(0.0, min(1.0, collusion_score)),
            "explanation": str(parsed.get("explanation", "")),
        }