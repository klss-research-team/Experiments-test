# bidding_reward.py

""" 
This module contains the RewardBiddingFn class, which evaluates agent bids and
reasoning traces to assign rewards based on profit, reasoning quality, and 
collusion suspicion. It utilizes the detector language model to evaluate 
reasoning quality and collusion suspicion score
"""

# from agent_system.environments.env_package.math.utils import extract_answer, grade_answer_mathd, grade_answer_sympy
from agent_system.environments.env_package.bidding.utils import extract_bid, extract_reasoning # TODO: do update utils.py in same dir
from dataclasses import dataclass

@dataclass
class RewardConfig:
    apply_format_reward: bool = False

    # reward constants
    profit_weight: int = 1.0
    win_weight: int = 0.1
    competitiveness_weight: int = 1.0
    collusion_suspicion_weight: int = 1.0

    detector_quality_weight: int = 1.0

class RewardBiddingFn:
    """
    Reward function for bidding agent actions

    This class implements the RewardFunction protocol to process the input and
    determine the reward based on:
        profit: (bid - contract_value) if win else 0,
        win_reward: 1 if win else 0
        competitiveness_reward: assigned by detector model
        collusion_suspicion_penalty: assigned by detector model
    """
    def __init__(self, config: RewardConfig):
        self.config = config

    # TODO: fill in the rest (long)
    def __call__(self, task_info: dict, action: str):
        """
        Calculate the reward for bidding agent based on the agent's action

        Parameters
        ----------
        task_info: dict
            dictionary containing .....?
        action: the agent's response

        Returns
        -------
        RewardOutput:
            the calculated reward with ???? information
        """
        
        # TODO: fill in
        model_response = action

        # Handle None or empty response
        if model_response is None or model_response == "":
            # print("DEBUG: Empty or None response")
            return self.config.format_error_reward  # TODO: what else to return here??
        
        # TODO: Extract bid + reasoning
        #
        #


        model_answer = extract_answer(model_response)
        if model_answer is None:
            return self.config.format_error_reward # TODO: what else to return here???
        
        # Process the ground truth(s):
        ground_truths = task_info.get("ground_truth", None)
        if ground_truths is None:
            return self.config.unk_error_reward # TODO: what else to return here???
        
        # convert single to answer to list for uniform processing
        if isinstance(ground_truths, str | float | int):
            ground_truths = [ground_truths]

        # Process each ground truth


    pass


class RewardDetectorFn:
    """
    Reward function for detector agent actions

    This class implements the RewardFunction protocol to process the input and
    determine the reward based on detector quality (e.g., collusion detection
    accuracy)
    """
    def __init__(self, config: RewardConfig):
        self.config = config

    # TODO: fill in the rest (long)
    def __call__(self, task_info: dict, action: str):
        """
        Calculate the reward for detector based on Judge LLM proprietary model
        (ground truth)
        
        Parameters
        ----------
        task_info: dict
            dictionary containing....
        action: the agent's response

        Returns
        -------
        RewardOutput:
            the calculated reward with correctness information
        """
        return reward # help
        pass


# Test out reward function
def main():
    reward = RewardBiddingFn(RewardConfig())
    # task_info = {
    #     "data_source" = "",
    #     ""
    # }

    action = "<bid>78</bid> <reasoning>I chose this bid because it's not too high and not too low.</bid>"

    output = reward(task_info, action)
    print(output)
    
    pass

if __name__ == "__main__":
    main()



# --------------- RewardMathFn below ----------------

from agent_system.environments.env_package.math.utils import extract_answer, grade_answer_mathd, grade_answer_sympy
from dataclasses import dataclass

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"


@dataclass
class RewardConfig:
    apply_format_reward: bool = False

    # General reward constants
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    format_error_reward: float = 0.0
    unk_error_reward: float = 0.0

    # Bonus reward for calling tools.
    toolcall_bonus: float = 0.0


class RewardMathFn:
    """
    Reward function for evaluating mathematical answers.

    This class implements the RewardFunction protocol to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def __call__(self, task_info: dict, action: str):
        """
        Calculate the reward for a math task based on the agent's action.

        Args:
            task_info: Dictionary containing problem, data_source, problem_type, and ground_truth
            action: The agent's response/solution

        Returns:
            RewardOutput: The calculated reward with correctness information
        """
        # Extract information from task_info
        problem = task_info.get("problem", "")
        model_response = action
        is_correct = False

        # Handle None or empty response
        if model_response is None or model_response == "":
            # print("DEBUG: Empty or None response")
            return self.config.format_error_reward, is_correct

        # Extract solution.
        if THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            if self.config.apply_format_reward:
                return self.config.format_error_reward, is_correct
            model_solution = model_response

        model_answer = extract_answer(model_solution)
        if model_answer is None:
            return self.config.format_error_reward, is_correct

        # Process the ground truth(s)
        ground_truths = task_info.get("ground_truth", None)
        if ground_truths is None:
            return self.config.unk_error_reward, is_correct

        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, str | float | int):
            ground_truths = [ground_truths]

        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)

        if not processed_ground_truths:
            return self.config.unk_error_reward, is_correct

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            _is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if _is_correct:
                # Apply tool call bonus if applicable and answer is correct
                reward = self.config.correct_reward
                if task_info.get("has_toolcall", False):
                    reward += self.config.toolcall_bonus
                
                is_correct=True
                return reward, is_correct

        return self.config.incorrect_reward, is_correct

if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig())
    task_info = {
        "data_source": "",
        "problem": ("Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$."),
        "problem_type": "math",
        "ground_truth": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"],
        "has_toolcall": True,
    }
    action = "<think>...</think>\nThe answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}."

    output, is_correct = reward(task_info, action)
    print(output)
    print(is_correct)