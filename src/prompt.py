THINKING_MODEL_SFT_PROMPT = """A conversation takes place between the user and the assistant. The user asks a question, and the assistant solves the problem. Please reason step by step, and put your final answer in the <answer> </answer> tags. Reasoning and answer must be in Bengali.

Question: {problem}
"""

INSTRUCT_MODEL_SFT_PROMPT = """A conversation takes place between the user and the assistant. The user asks a question, and the assistant solves the problem. Please reason step by step, and put your reasoning process and final answer in the <think> </think> and <answer> </answer> tags respectively. Reasoning and answer must be in Bengali. Final answer must be a number nothing else.

Question: {problem}
"""


THINKING_MODEL_RESPONSE = (
    """<think>\n{cot}\n</think>\n\n<answer>\n{answer}\n</answer>"""
)

INSTRUCT_MODEL_RESPONSE = (
    """<think>\n{cot}\n</think>\n\n<answer>\n{answer}\n</answer>"""
)
