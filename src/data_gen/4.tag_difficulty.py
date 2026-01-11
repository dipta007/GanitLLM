import argparse
import copy
import logging
import os
import re
import random

from datasets import load_dataset, concatenate_datasets
from joblib import Parallel, delayed
from openai import OpenAI
from retry import retry
from tqdm import tqdm, trange

from src.evaluation import common
from src.evaluation.vllm_utils import kill_vllm, run_vllm

logger = logging.getLogger(__name__)

# random port
port = random.randint(8888, 9999)
print(f"Using port {port}")

client = OpenAI(
    base_url=f"http://localhost:{port}/v1",
    api_key="empty",
)

INSTRUCTIONS_TEMPLATE = """এই গণিতের সমস্যাটি সমাধান করুন। চূড়ান্ত উত্তর দেওয়ার আগে যুক্তিসম্পন্ন পদক্ষেপ প্রদান করুন। চূড়ান্ত উত্তরটি একক সংখ্যা হিসাবে "উত্তর:" এর পরে শেষ লাইনে দিন। "উত্তর:" এর পরে অন্য কিছু যুক্ত করবেন না।.

{input}"""


ANSWER_PREFIX = "উত্তর"


def parse_answer(answer: str, answer_prefix: str) -> str:
    if not answer:
        return ""

    if answer_prefix not in answer:
        return ""

    answer_text = answer.split(answer_prefix)[-1].strip()

    # find all the numbers (including decimals) in the string
    numbers = re.findall(r"\d+\.?\d*", answer_text.replace(",", ""))

    # return the first number (removing trailing decimal point if present),
    # or an empty string if there were no numbers
    return numbers[-1].rstrip(".") if numbers else ""


def get_score(target: str, prediction: str) -> bool:
    target = str(target)
    prediction = str(prediction)
    target = common.normalize_text(target)
    prediction = common.normalize_text(prediction)

    return target == prediction


def get_difficulty(correct_count):
    # bucket of 8 for correct_counts
    if correct_count <= 8:
        return "olympiad"
    elif correct_count <= 16:
        return "hard"
    elif correct_count <= 24:
        return "medium"
    else:
        return "easy"


def process_row(args, row):
    @retry(tries=-1, delay=1, backoff=2, max_delay=2, logger=logger)
    def get_response():
        response = client.chat.completions.create(
            model=args.model_name,
            messages=[
                {
                    "role": "user",
                    "content": INSTRUCTIONS_TEMPLATE.format(
                        input=row[args.problem_column]
                    ),
                }
            ],
            max_tokens=2400,
            n=args.num_attempts,
        )
        print(f"Response: {response.usage}")
        return response

    response = get_response()

    curr_generated_solutions = []
    curr_prompts = []
    correct_count = 0
    for output in response.choices:
        generated_solution = output.message.content
        parsed_generated_solution = parse_answer(generated_solution, ANSWER_PREFIX)
        correct_count += get_score(parsed_generated_solution, row[args.solution_column])

        curr_generated_solutions.append(generated_solution)
        curr_prompts.append(
            INSTRUCTIONS_TEMPLATE.format(input=row[args.problem_column])
        )

    difficulty = get_difficulty(correct_count)
    return curr_generated_solutions, correct_count, difficulty, curr_prompts


def main(args):
    # assert args.hf_repo_id is not None, "hf_repo_id is required"

    # Load the dataset
    dataset = load_dataset(args.dataset_name)["train"]

    processed_dataset = None
    try:
        processed_dataset = load_dataset(f"{args.dataset_name}_tagged_difficulty")[
            "train"
        ]
        processed_ids = set(processed_dataset["id"])
        print(f"Loaded {len(processed_ids)} processed rows")
        prev_len = len(dataset)
        dataset = dataset.filter(lambda x: x["id"] not in processed_ids)
        print(f"Filtered out {prev_len - len(dataset)} rows")
    except Exception as e:
        print(f"Error loading processed dataset: {e}")
        pass

    print(f"Processing {len(dataset)} rows")

    if args.debug:
        print("Debug mode: using 400 examples")
        dataset = dataset.select(range(200))

    difficulties = []
    correct_counts = []
    generated_solutions = []
    prompts = []
    for i in trange(0, len(dataset), args.batch_size):
        start, end = i, min(i + args.batch_size, len(dataset))
        batch = dataset.select(range(start, end))

        print(f"Generating solutions for batch {i} of {len(dataset)}")
        outputs = Parallel(n_jobs=args.batch_size, backend="threading")(
            delayed(process_row)(args, row) for row in tqdm(batch)
        )
        (
            curr_generated_solutions,
            curr_correct_counts,
            curr_difficulties,
            curr_prompts,
        ) = zip(*outputs)

        generated_solutions.extend(curr_generated_solutions)
        correct_counts.extend(curr_correct_counts)
        difficulties.extend(curr_difficulties)
        prompts.extend(curr_prompts)

        copy_dataset = copy.deepcopy(dataset)
        copy_dataset = copy_dataset.select(range(0, end))
        copy_dataset = copy_dataset.add_column(
            "generated_solutions", generated_solutions
        )
        copy_dataset = copy_dataset.add_column("prompts", prompts)
        copy_dataset = copy_dataset.add_column("correct_counts", correct_counts)
        copy_dataset = copy_dataset.add_column("difficulty", difficulties)
        if processed_dataset is not None:
            copy_dataset = concatenate_datasets([copy_dataset, processed_dataset])
            print(f"Merged {len(processed_dataset)} rows, now {len(copy_dataset)} rows")
        copy_dataset.push_to_hub(f"{args.dataset_name}_tagged_difficulty")
        print(
            f"Successfully pushed the dataset at index {i} to {args.dataset_name}_tagged_difficulty"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve math problems and determine difficulty."
    )
    parser.add_argument(
        "--dataset_name",
        "-d",
        type=str,
        required=True,
        help="Name of the Hugging Face dataset.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Name of the vLLM model.",
    )
    parser.add_argument(
        "--problem_column",
        type=str,
        default="problem",
        help="Name of the column with problems.",
    )
    parser.add_argument(
        "--solution_column",
        type=str,
        default="bengali_solution",
        help="Name of the column with solutions.",
    )
    parser.add_argument(
        "--num_attempts",
        type=int,
        default=32,
        help="Number of times to attempt each problem.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for processing."
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode.")

    args = parser.parse_args()
    import time

    start_time = time.time()
    proc = run_vllm(args.model_name, port)
    main(args)
    kill_vllm([proc])
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
