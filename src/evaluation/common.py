import io
import os
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from typing import Any, Callable

import numpy as np
import requests
from tqdm import tqdm

from .eval_types import EvalResult, Message, SingleEvalResult


def _compute_stat(values: list, stat: str):
    if stat == "mean":
        return np.mean(values)
    elif stat == "std":
        return np.std(values)
    elif stat == "min":
        return np.min(values)
    elif stat == "max":
        return np.max(values)
    elif stat == "n_samples":
        return len(values)
    elif stat == "bootstrap_std":
        return np.std(
            [np.mean(np.random.choice(values, len(values))) for _ in range(1000)]
        )
    else:
        raise ValueError(f"Unknown {stat =}")


def aggregate_results(
    single_eval_results: list[SingleEvalResult],
    default_stats: tuple[str, ...] = ("mean", "std"),
    name2stats: dict[str, tuple[str]] | None = None,
) -> EvalResult:
    """
    Aggregate results from multiple evaluations into a single EvalResult.
    """
    name2stats = name2stats or {}
    name2values = defaultdict(list)
    convos = []
    correct_answers = []
    extracted_answers = []
    metadata = []
    finish_reasons = []
    for single_eval_result in single_eval_results:
        for name, value in single_eval_result.metrics.items():
            name2values[name].append(value)
        if single_eval_result.score is not None:
            name2values["score"].append(single_eval_result.score)
        convos.append(single_eval_result.convo)
        correct_answers.append(single_eval_result.correct_answer)
        extracted_answers.append(single_eval_result.extracted_answer)
        metadata.append(single_eval_result.example_level_metadata)
        finish_reasons.append(single_eval_result.finish_reason)
    final_metrics = {}
    for name, values in name2values.items():
        stats = name2stats.get(name, default_stats)
        for stat in stats:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_stat(values, stat)
    return EvalResult(
        score=final_metrics.pop("score", None),
        metrics=final_metrics,
        convos=convos,
        correct_answers=correct_answers,
        extracted_answers=extracted_answers,
        finish_reasons=finish_reasons,
        metadata={"example_level_metadata": metadata},
    )


def map_with_progress(
    f: Callable,
    xs: list[Any],
    num_threads: int = 128,
    pbar: bool = True,
):
    """
    Apply f to each element of xs, using a ThreadPool, and show progress.
    """
    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x

    if os.getenv("debug"):
        return list(map(f, pbar_fn(xs, total=len(xs))))
    else:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            return list(pbar_fn(pool.imap(f, xs), total=len(xs)))


def convert_solution_to_bengali(text):
    processed_solution = ""
    for char in text:
        if char in "0123456789":
            mapping = {
                "0": "০",
                "1": "১",
                "2": "২",
                "3": "৩",
                "4": "৪",
                "5": "৫",
                "6": "৬",
                "7": "৭",
                "8": "৮",
                "9": "৯",
            }
            processed_solution += mapping[char]
        elif char in "০১২৩৪৫৬৭৮৯":
            processed_solution += char
        elif char in ".":
            processed_solution += "."
        elif char in ",":
            pass
        else:
            print(f"error: {text}")
            return ""
            # raise ValueError(f"error: {char}")
    return processed_solution


def convert_solution_to_english(text):
    processed_solution = ""
    for char in text:
        if char in "০১২৩৪৫৬৭৮৯":
            mapping = {
                "০": "0",
                "১": "1",
                "২": "2",
                "৩": "3",
                "৪": "4",
                "৫": "5",
                "৬": "6",
                "৭": "7",
                "৮": "8",
                "৯": "9",
            }
            processed_solution += mapping[char]
        elif char in "0123456789":
            processed_solution += char
        elif char in ".":
            processed_solution += "."
        elif char in ", ":
            pass
        else:
            print(f"error: *{text}*")
            return ""
            # raise ValueError(f"error: {char}")

    return processed_solution


def normalize_text(text: str) -> str:
    if "." in text:
        text = text.rstrip("0").rstrip("০").rstrip(".")
    text = convert_solution_to_english(text)
    text = text.replace(",", "")
    return text


def url_to_fileobj(url: str, binary=False) -> Any:
    response = requests.get(url)
    response.raise_for_status()
    return io.BytesIO(response.content) if binary else io.StringIO(response.text)


def has_only_user_assistant_messages(messages: list[Message]) -> bool:
    """
    Check if the messages only contain user and assistant messages.
    """
    return all(m["role"] in ("user", "assistant") for m in messages)
