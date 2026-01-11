import logging
import json
import os
from collections import Counter

from joblib import Parallel, delayed
from datasets import load_dataset
from openai import OpenAI
from retry import retry
from tqdm import tqdm

from src.evaluation import common


SYSTEM_PROMPT = """You are an expert mathematics assistant. A conversation takes place between the user and the assistant. The user asks a question, and the assistant solves the mathematical problem. The assistant first thinks through the reasoning process in their mind and then provides the answer to the user. The reasoning process and the answer are enclosed within <think> </think> and <answer> </answer> tags respectively.

Your answer must be in the following format:
<think>
Reasoning process here. Please reason step by step. It can be long and detailed.
</think>
<answer>
Answer here in a single number, do not add any other text, other than a single number.
</answer>.

Question: {problem}
"""


logger = logging.getLogger(__name__)


# @retry(tries=-1, delay=64, backoff=2, max_delay=64, logger=logger)
def completion_with_backoff(client, model, prompt):
    assert "gpt" not in model, "gpt is finished"
    assert "grok" not in model, "grok is finished"
    assert "gemini" not in model, "gemini is finished"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="medium",
            max_tokens=16384,
        )
        assert response.choices[0].finish_reason == "stop", "finish reason not stop"
    except Exception as e:
        print(f"Error: {e}")
        raise e

    print("got response")
    return response.model_dump()


def validate(response, gt_solution):
    def parse_answer_with_think(answer: str) -> str:
        if answer.count("<answer>") == 1 and answer.count("</answer>") == 1:
            number = answer.split("<answer>")[1].split("</answer>")[0].strip()
            return number.rstrip(".") if number else ""
        else:
            raise ValueError(f"token format not correct: {answer}")

    pred_solution = parse_answer_with_think(response)

    target = common.normalize_text(gt_solution)
    pred_solution = common.normalize_text(pred_solution)

    if pred_solution == "":
        raise ValueError("Pred Solution empty")

    return target == pred_solution, target, pred_solution


def is_valid(outputs):
    # count = Counter(outputs)
    # return count.most_common(1)[0][0]
    if any(output == -1 for output in outputs):
        return -1
    elif all(output == 1 for output in outputs):
        return 1
    else:
        return 0


def run_row(row, turn, client, model, dir_name):
    try:
        if os.path.exists(f"{dir_name}/{turn}/{row['id']}.json"):
            with open(
                f"{dir_name}/{turn}/{row['id']}.json", "r", encoding="utf-8"
            ) as f:
                data = json.load(f)
                response = data["response"]
        else:
            response = completion_with_backoff(
                client, model, SYSTEM_PROMPT.format(problem=row["problem"])
            )

        output = response["choices"][0]["message"]["content"]
        valid, target, pred_solution = validate(output, row["english_solution"])

        data = {
            "valid": valid,
            "response": response,
            "turn": turn,
            "row": row,
            "output": output,
            "gt_solution": row["english_solution"],
            "pred_solution": pred_solution,
            "target": target,
        }
        with open(f"{dir_name}/{turn}/{row['id']}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error without saving: {e}")


def get_results(args, ds, client, model):
    dir_name = f"{model.replace('/', '_')}_responses"
    outputs = [-1] * len(ds)

    func_args = []
    for i, row in tqdm(enumerate(ds), total=len(ds), desc="Getting Arguments"):
        for turn in range(args.num_turns):
            os.makedirs(f"{dir_name}/{turn}", exist_ok=True)
            func_args.append((row, turn, client, model, dir_name))

    Parallel(n_jobs=32, backend="threading")(
        delayed(run_row)(*func_args)
        for func_args in tqdm(func_args, total=len(func_args), desc="Running Rows")
    )

    for i, row in tqdm(enumerate(ds), total=len(ds), desc="Getting Results"):
        turn_results = []
        for turn in range(args.num_turns):
            if os.path.exists(f"{dir_name}/{turn}/{row['id']}.json"):
                with open(
                    f"{dir_name}/{turn}/{row['id']}.json", "r", encoding="utf-8"
                ) as f:
                    data = json.load(f)
                turn_results.append(int(data["valid"]))
            else:
                turn_results.append(-1)
        # get the majority vote
        count = Counter(turn_results)
        outputs[i] = count.most_common(1)[0][0]

    return outputs


def validate_by_gpt(args, ds):
    print("GPT model: ", args.gpt)
    print("GPT API key: ", os.getenv("OPENAI_API_KEY"))
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    outputs = get_results(args, ds, client, args.gpt)
    return outputs


def validate_by_grok(args, ds):
    print("Grok model: ", args.grok)
    print("Grok API key: ", os.getenv("OPENROUTER_API_KEY"))
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    outputs = get_results(args, ds, client, args.grok)
    return outputs


def validate_by_gemini(args, ds):
    print("Gemini model: ", args.gemini)
    print("Gemini API key: ", os.getenv("OPENROUTER_API_KEY"))
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    outputs = get_results(args, ds, client, args.gemini)

    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt", type=str, default="gpt-5-mini")
    parser.add_argument("--gemini", type=str, default="google/gemini-2.5-flash")
    parser.add_argument("--grok", type=str, default="x-ai/grok-4-fast:free")
    parser.add_argument("--num_turns", type=int, default=3)
    args = parser.parse_args()

    ds = load_dataset("GanitLLM/Ganit_curriculum_shuffle", "dev")["dev"]
    print(ds)

    for col in ["gpt_outputs", "gemini_outputs", "grok_outputs", "valid"]:
        if col in ds.column_names:
            ds = ds.remove_columns([col])

    print("========================================")
    gpt_outputs = validate_by_gpt(args, ds)
    ds = ds.add_column("gpt_outputs", gpt_outputs)
    print("GPT outputs: ", Counter(gpt_outputs))
    print("========================================")
    gemini_outputs = validate_by_gemini(args, ds)
    ds = ds.add_column("gemini_outputs", gemini_outputs)
    print("Gemini outputs: ", Counter(gemini_outputs))
    print("========================================")
    grok_outputs = validate_by_grok(args, ds)
    ds = ds.add_column("grok_outputs", grok_outputs)
    print("Grok outputs: ", Counter(grok_outputs))
    print("========================================")

    valid = []
    for i in range(len(ds)):
        valid.append(
            is_valid(
                [ds["gpt_outputs"][i], ds["gemini_outputs"][i], ds["grok_outputs"][i]]
            )
        )

    print("Valid: ", valid)
    ds = ds.add_column("valid", valid)
    print("counter", Counter(valid))
    print(ds)

    ds.push_to_hub("GanitLLM/Ganit_curriculum_shuffle", "dev")


# 15 million tokens for 3 runs each model
