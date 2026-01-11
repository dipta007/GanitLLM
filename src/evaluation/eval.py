import argparse
import os
import re

import jsonlines
import torch
from datasets import load_dataset

from src.evaluation import common
from src.evaluation.eval_types import EvalResult, SingleEvalResult
from src.evaluation.vllm_utils import run_vllm, sampler
from src.prompt import THINKING_MODEL_SFT_PROMPT, INSTRUCT_MODEL_SFT_PROMPT

DATASET_TO_HF = {
    "mgsm": "jbross-ibm-research/mgsm",
    "msvamp": "Mathoctopus/MSVAMP",
}


def parse_answer_with_think(answer: str) -> str:
    # if answer.count("<think>") == 1 and answer.count("</think>") == 1 and answer.count("<answer>") == 1 and answer.count("</answer>") == 1:
    # if answer.count("<answer>") == 1 and answer.count("</answer>") == 1:
    matches = re.findall(r"<answer>(.*?)</answer>", answer, re.DOTALL)
    if len(matches) > 0:
        number = matches[-1].strip()
        number = number.rstrip(".") if number else ""
        return number
    else:
        print(f"Error: {answer}")
        return ""


def get_score(target: str, prediction: str) -> bool:
    target = common.normalize_text(target)
    prediction = common.normalize_text(prediction)

    return target == prediction


def get_dataset(dataset: str) -> list[dict[str, str]]:
    if dataset in ["ganit_easy", "ganit_medium", "ganit_hard", "ganit_olympiad"]:
        ds = []
        difficulty = dataset.split("_")[-1]
        with jsonlines.open(f"data_v2/{difficulty}_dev.jsonl") as reader:
            for line in reader:
                ds.append(
                    {
                        "inputs": line["problem"],
                        "targets": line["bengali_solution"],
                    }
                )
    else:
        ds = load_dataset(DATASET_TO_HF[dataset], "bn", split="test")
    if dataset == "mgsm":
        ds = ds.rename_column("question", "inputs")
        ds = ds.rename_column("answer_number", "targets")
    elif dataset == "msvamp":
        ds = ds.rename_column("m_query", "inputs")
        ds = ds.rename_column("response", "targets")

    examples = []
    for example in ds:
        examples.append(
            {
                "inputs": example["inputs"],
                "targets": str(example["targets"]),
                "lang": "bn",
            }
        )

    if os.getenv("debug"):
        print(f"Debug mode: using {len(examples[:100])} examples")
        return examples[:100]
    else:
        return examples


class MathEval:
    def __init__(self, dataset: str, port: int, model_id: str):
        self.port = port
        self.model_id = model_id
        self.examples = get_dataset(dataset)

    def __call__(self, sampler) -> EvalResult:
        def fn(example: dict[str, str]):
            language = example["lang"]
            correct_answer = example["targets"]

            prompt_template = THINKING_MODEL_SFT_PROMPT
            if "tiger" in self.model_id.lower():
                prompt_template = INSTRUCT_MODEL_SFT_PROMPT
            if "llama" in self.model_id.lower():
                prompt_template = INSTRUCT_MODEL_SFT_PROMPT
            if "gpt" in self.model_id.lower():
                prompt_template = INSTRUCT_MODEL_SFT_PROMPT
            prompt_messages = [
                dict(
                    content=prompt_template.format(problem=example["inputs"]),
                    role="user",
                ),
            ]
            try:
                sampler_response = sampler(prompt_messages, self.port, self.model_id)
                response_text = sampler_response.response_text
                actual_queried_prompt_messages = (
                    sampler_response.actual_queried_message_list
                )
                finish_reason = sampler_response.finish_reason
            except Exception as e:
                response_text = ""
                actual_queried_prompt_messages = []
                finish_reason = None

            extracted_answer = parse_answer_with_think(response_text)

            score = get_score(correct_answer, extracted_answer)
            convo = actual_queried_prompt_messages + [
                dict(content=response_text, role="assistant")
            ]
            return SingleEvalResult(
                score=score,
                convo=convo,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
                finish_reason=finish_reason,
                metrics={language: score},
            )

        num_threads = 128
        if any(
            arg in self.model_id.lower()
            for arg in ["14", "32", "70", "72", "120", "235"]
        ):
            num_threads = 32
        results = common.map_with_progress(fn, self.examples, num_threads=num_threads)
        return common.aggregate_results(results, default_stats=("mean", "std"))


def run_eval_with_vllm(dataset: str, model_id: str):
    port = 8888
    device_count = torch.cuda.device_count()
    devices = ",".join(str(i) for i in range(device_count))

    proc = run_vllm(model_id, port, devices)
    eval = MathEval(dataset=dataset, port=port, model_id=model_id)
    eval_result = eval(sampler)
    # kill_vllm([proc])
    # print("VLLM killed")
    return dict(
        score=eval_result.score,
        metrics=eval_result.metrics,
        convos=eval_result.convos,
        metadata=eval_result.metadata,
        correct_answers=eval_result.correct_answers,
        extracted_answers=eval_result.extracted_answers,
        model_id=model_id,
        dataset=dataset,
    )


def run_eval(dataset: str, model_id: str, port: int):
    eval = MathEval(dataset=dataset, port=port, model_id=model_id)
    eval_result = eval(sampler)
    return dict(
        score=eval_result.score,
        metrics=eval_result.metrics,
        convos=eval_result.convos,
        metadata=eval_result.metadata,
        correct_answers=eval_result.correct_answers,
        extracted_answers=eval_result.extracted_answers,
        finish_reasons=eval_result.finish_reasons,
        model_id=model_id,
        dataset=dataset,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["mgsm", "msvamp"]
    )
    args = parser.parse_args()

    eval_result = run_eval_with_vllm(args.dataset, args.model_name)
    print(eval_result["model_id"])
    print(eval_result["dataset"])
    print(eval_result["score"])
