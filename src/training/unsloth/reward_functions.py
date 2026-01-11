import os
import string
from typing import List
from tqdm import tqdm


def soft_overlong_reward(
    completions, soft_limit=2000, hard_limit=2500, **kwargs
) -> List[float]:
    """
    Soft overlong reward function.
    """

    rewards = []
    for completion in completions:
        completion = completion[0]["content"]
        current_tokens = (len(completion.split()) * 1000) // 750

        if current_tokens > hard_limit:
            rewards.append(-1.0)
        elif current_tokens > soft_limit:
            length_difference = current_tokens - soft_limit
            reward = length_difference / (hard_limit - soft_limit)
            rewards.append(-reward)
        else:
            rewards.append(0.0)

        print(f"Current tokens: {current_tokens}, Reward: {rewards[-1]}")
    return rewards


def response_one_reward_rules_all(completions, **kwargs) -> List[float]:
    """
    Unified reward function that checks format, Bengali reasoning, and answer correctness.

    Args:
        completions: List of model completions
        **kwargs: Should contain 'bengali_solution' and 'english_solution' lists

    Returns:
        List of reward scores (0.0 to 2.0)
    """
    max_possible_reward = 2.0
    min_possible_reward = 0.0

    def is_correct_format(completion: str) -> bool:
        if (
            completion.count("<think>") == 1
            and completion.count("</think>") == 1
            and completion.count("<answer>") == 1
            and completion.count("</answer>") == 1
        ):
            try:
                text = completion.split("<think>")[-1]
                text = text.split("</think>")[-1]
                text = text.split("<answer>")[-1]
                text = text.split("</answer>")[-1]
                return True
            except:
                return False
        return False

    def is_enough_bengali_in_reasoning(completion: str, threshold: float = 0.8) -> bool:
        if os.getenv("BENGALI_REASONING_REWARD", "False").lower() == "false":
            return True

        reasoning = completion.split("<think>")[-1].split("</think>")[0].strip()
        valid_chars = set(
            "অআইঈউঊঋঌএঐঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ঺঻়ঽািীুূৃৄেৈোৌ্ৎৗড়ঢ়য়ৠৡৢৣ০১২৩৪৫৬৭৮৯।ওংঃ×÷"
            + string.punctuation
        )
        bengali_chars = sum(1 for char in reasoning if char in valid_chars)
        total_chars = sum(1 for char in reasoning if char != " ")
        bengali_percentage = bengali_chars / max(total_chars, 1)
        return bengali_percentage > threshold

    def normalize_text(completion: str) -> str:
        """Normalize answer text for comparison."""
        if "." in completion:
            completion = completion.rstrip("0").rstrip("০").rstrip(".")
        completion = completion.replace(",", "")
        return completion

    def is_correct_answer(completion: str, bn_solution: str, en_solution: str) -> float:
        gen_response = completion.split("<answer>")[1].split("</answer>")[0].strip()
        gen_response = normalize_text(gen_response)
        gt_bengali_response = normalize_text(bn_solution)
        gt_english_response = normalize_text(en_solution)

        reward = min_possible_reward
        if gen_response == gt_bengali_response:
            reward = max_possible_reward
        elif gen_response == gt_english_response:
            reward = max_possible_reward / 2
        return reward

    # Final rewards calculation
    rewards = []
    for completion, bn_solution, en_solution in tqdm(
        zip(completions, kwargs["bengali_solution"], kwargs["english_solution"]),
        total=len(completions),
    ):
        completion = completion[0]["content"]
        reward = min_possible_reward
        if is_correct_format(completion) and is_enough_bengali_in_reasoning(completion):
            reward = is_correct_answer(completion, bn_solution, en_solution)

        rewards.append(reward)

    return rewards
