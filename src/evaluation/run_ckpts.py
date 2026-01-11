import json
import os
import random
import re
import string
import time

from transformers import AutoTokenizer

import wandb
from src.evaluation.eval import run_eval
from src.evaluation.vllm_utils import kill_vllm, run_vllm

VERSION = "v15"

base_models = [
    # "md-nishat-008/TigerLLM-9B-it",
    # "md-nishat-008/TigerLLM-1B-it",
    # "dipta007/TigerLLM-1B-it",
    # "Qwen/Qwen2.5-0.5B-Instruct",
    # "Qwen/Qwen2.5-1.5B-Instruct",
    # "Qwen/Qwen2.5-3B-Instruct",
    # "Qwen/Qwen2.5-7B-Instruct",
    # "Qwen/Qwen2.5-14B-Instruct",
    # "Qwen/Qwen2.5-32B-Instruct",
    # "Qwen/Qwen3-0.6B",
    # "Qwen/Qwen3-1.7B",
    # "Qwen/Qwen3-4B",
    # "Qwen/Qwen3-8B",
    # "Qwen/Qwen3-14B",
    # "Qwen/Qwen3-32B",
    # "Qwen/Qwen3-4B-Instruct-2507",
    # "Qwen/Qwen3-4B-Thinking-2507",
    # "meta-llama/Llama-3.2-1B-Instruct",
    # "meta-llama/Llama-3.2-3B-Instruct",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "meta-llama/Llama-3.3-70B-Instruct",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
    # "gpt-5.1-mini",
    # "gpt-5.1",
]


ROOT_SFT_DIR = "/umbc/rs/pi_ferraro/users/sroydip1/GanitLLM_checkpoints/sft_v4"
sft_dirs = []
for d in os.listdir(ROOT_SFT_DIR):
    continue
    print(d)
    sft_dirs.append(os.path.join(ROOT_SFT_DIR, d))

ROOT_GRPO_DIR = "/umbc/rs/pi_ferraro/users/sroydip1/GanitLLM_checkpoints/grpo_v5"
grpo_dirs = []
for d in os.listdir(ROOT_GRPO_DIR):
    print(d)
    grpo_dirs.append(os.path.join(ROOT_GRPO_DIR, d))

ROOT_GRPO_DIR = "/umbc/rs/pi_ferraro/users/sroydip1/GanitLLM_checkpoints/grpo_v6"
for d in os.listdir(ROOT_GRPO_DIR):
    print(d)
    grpo_dirs.append(os.path.join(ROOT_GRPO_DIR, d))

# exit()

checkpoint_dirs = []
for dir in sft_dirs + grpo_dirs:
    continue
    if not os.path.exists(dir):
        continue
    checkpoint_dirs.append(dir)
    # for d in os.listdir(dir):
    #     checkpoint_dirs.append(os.path.join(dir, d))
    # if d.startswith("v"):
    #     checkpoint_dirs.append(os.path.join(dir, d))
dirs = base_models + checkpoint_dirs
dirs = dirs[::-1]
random.shuffle(dirs)


datasets = [
    "mgsm",
    "msvamp",
    "ganit_easy",
    "ganit_medium",
    "ganit_hard",
    "ganit_olympiad",
]

PORT = random.randint(8888, 9999)


def get_percentage_of_bengali(convos):
    def calculate_bengali_percentage(question):
        question = re.sub(r"\s+", "", question)
        bengali_chars = 0
        total_chars = len(question)
        for char in question:
            if (
                char
                in "অআইঈউঊঋঌএঐঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ঺঻়ঽািীুূৃৄেৈোৌ্ৎৗড়ঢ়য়ৠৡৢৣ০১২৩৪৫৬৭৮৯।ওংঃ×÷"
                + string.punctuation
            ):
                bengali_chars += 1
        return bengali_chars / max(total_chars, 1)

    char_percentage = []
    for convo in convos:
        char_percentage.append(calculate_bengali_percentage(convo[-1]["content"]))
    return sum(char_percentage) / len(convos)


def number_of_words(convos, model_id):
    words = []
    for convo in convos:
        convo = convo[-1]["content"]
        # replace multiple spaces with single space
        convo = re.sub(r"\s+", " ", convo)
        words.append(len(convo.split()))

    return sum(words) / len(words)


def number_of_tokens(convos, model_id):
    return 0
    questions = []
    for convo in convos:
        questions.append(convo[-1]["content"])

    tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(questions, return_tensors="pt", padding="longest")
    attention_mask = tokens.attention_mask.cpu()
    return attention_mask.sum().item() / len(questions)


def find_wandb_run(model_id, version):
    runs = wandb.Api().runs(
        "collab-srd/GanitLLM",
        filters={"config.model_id": model_id, "config.version": version},
    )
    return runs[0].id if runs else None


def get_base_and_lora_path(lora_path, ckpt_num):
    if ckpt_num > 0:
        try:
            with open(os.path.join(lora_path, "adapter_config.json"), "r") as f:
                args = json.load(f)
                base_model = args["base_model_name_or_path"]
                if "unsloth" in base_model.lower():
                    base_model = base_model.replace("unsloth/", "Qwen/")
                train_type = args["peft_type"].lower()
        except Exception as e:
            with open(os.path.join(lora_path, "args.json"), "r") as f:
                args = json.load(f)
                base_model = args["model"]
                if "unsloth" in base_model.lower():
                    base_model = base_model.replace("unsloth/", "Qwen/")
                train_type = args["train_type"].lower()

        if train_type == "full":
            base_model = lora_path
            lora_path = None
    else:
        base_model = lora_path
        lora_path = None
    return base_model, lora_path


def run_all_evaluations():
    print("Starting evaluations...")
    print(f"Dirs: {len(dirs)}")
    print(f"Datasets: {len(datasets)}")
    print("=" * 50)

    for dir in dirs:
        curr_model_ids = []
        if os.path.exists(dir):
            curr_run_name = dir.split("/")[-1]
            if "grpo" in dir.lower():
                curr_run_name = "grpo_" + curr_run_name
            for d in os.listdir(dir):
                print(d)
                if d.startswith("checkpoint-"):
                    ckpt_num = int(d.split("-")[-1])
                    curr_model_ids.append((os.path.join(dir, d), ckpt_num))

            random.shuffle(curr_model_ids)
        else:
            ckpt_num = 0
            curr_model_ids.append((dir, ckpt_num))
            curr_run_name = dir.split("/")[-1]

        print(dir)
        print(curr_run_name)
        print(curr_model_ids)
        print("=" * 50)

        wandb_run_id = find_wandb_run(curr_run_name, VERSION)
        wandb_init_config = dict(
            project="GanitLLM",
            entity="collab-srd",
            name=curr_run_name,
            config=dict(
                model_id=curr_run_name,
                run_name=curr_run_name,
                version=VERSION,
            ),
        )
        if wandb_run_id is not None:
            print(f"Resuming wandb run {wandb_run_id}")
            wandb_init_config["id"] = wandb_run_id
            wandb_init_config["resume"] = "must"
        run = wandb.init(**wandb_init_config)

        best_score = 0
        best_ckpt = None
        best_results = None
        for lora_path, ckpt_num in curr_model_ids:
            if int(ckpt_num) % 100 != 0:
                continue
            base_model, lora_path = get_base_and_lora_path(lora_path, ckpt_num)
            procs = []
            total_tokens = 0
            total_words = 0
            total_time_taken = 0
            total_bengali_percentage = 0
            total_score = 0
            for dataset in datasets:
                run.define_metric(step_metric="checkpoint", name=f"{dataset}_score")
                run.define_metric(
                    step_metric="checkpoint", name=f"{dataset}_num_of_tokens"
                )
                run.define_metric(
                    step_metric="checkpoint", name=f"{dataset}_num_of_words"
                )
                run.define_metric(
                    step_metric="checkpoint", name=f"{dataset}_percentage_of_bengali"
                )

                output_path = f"outputs/evaluation_{VERSION}/{curr_run_name}/{ckpt_num}/{dataset}.json"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                print(f"\nEvaluating {curr_run_name} on {dataset} at {ckpt_num}...")

                try:
                    eval_result = json.load(open(output_path))
                except Exception as e:
                    print(f"&&&& Error loading eval result: {e}")
                    # proc, call_model_id = run_vllm(base_model, PORT, lora_path=lora_path)
                    # procs.append(proc)
                    call_model_id = base_model
                    start_time = time.time()
                    eval_result = run_eval(dataset, call_model_id, PORT)
                    end_time = time.time()
                    eval_result["time_taken"] = end_time - start_time
                    if "num_of_tokens" not in eval_result["metrics"]:
                        eval_result["metrics"]["num_of_tokens"] = number_of_tokens(
                            eval_result["convos"], base_model
                        )
                    if "num_of_words" not in eval_result["metrics"]:
                        eval_result["metrics"]["num_of_words"] = number_of_words(
                            eval_result["convos"], base_model
                        )
                    if "percentage_of_bengali" not in eval_result["metrics"]:
                        eval_result["metrics"]["percentage_of_bengali"] = (
                            get_percentage_of_bengali(eval_result["convos"])
                        )

                    with open(output_path, "w") as f:
                        json.dump(eval_result, f, indent=2, ensure_ascii=False)
                    print(f"Results saved to: {output_path}")

                total_time_taken += eval_result["time_taken"]
                total_tokens += eval_result["metrics"]["num_of_tokens"]
                total_words += eval_result["metrics"]["num_of_words"]
                total_bengali_percentage += eval_result["metrics"][
                    "percentage_of_bengali"
                ]
                total_score += eval_result["score"]

                if ckpt_num == 0:
                    # for base model constant line
                    start = 50
                    end = 10001
                    step = 50
                else:
                    start = ckpt_num
                    end = ckpt_num + 1
                    step = 1

                for c in range(start, end, step):
                    run.log(
                        {
                            f"{dataset}_num_of_tokens": eval_result["metrics"][
                                "num_of_tokens"
                            ],
                            f"{dataset}_num_of_words": eval_result["metrics"][
                                "num_of_words"
                            ],
                            f"{dataset}_percentage_of_bengali": eval_result["metrics"][
                                "percentage_of_bengali"
                            ],
                            f"{dataset}_score": eval_result["score"],
                            f"{dataset}_time_taken": eval_result["time_taken"],
                            "checkpoint": c,
                        },
                    )
                print(f"✓ Completed: {curr_run_name} on {dataset} at {ckpt_num}")

            logged_data = {
                "total_tokens": total_tokens,
                "total_time_taken": total_time_taken,
                "total_words": total_words,
                "total_bengali_percentage": total_bengali_percentage,
                "mean_tokens": total_tokens / len(datasets),
                "mean_words": total_words / len(datasets),
                "mean_time_taken": total_time_taken / len(datasets),
                "mean_bengali_percentage": total_bengali_percentage
                / len(datasets)
                * 100,
                "mean_score": total_score / len(datasets),
                "checkpoint": c,
            }
            for c in range(start, end, step):
                run.log(logged_data)

            if total_score > best_score:
                best_score = total_score
                best_ckpt = c
                best_results = logged_data.copy()

            if len(procs) > 0:
                kill_vllm(procs)
                print(f"✓ VLLM for {curr_run_name} is killed")

        assert best_results is not None, (
            f"Best results are not found for {curr_run_name}"
        )
        print(f"✓ Best score: {best_score} at {best_ckpt}")
        best_output_path = f"outputs/evaluation_{VERSION}/{curr_run_name}/_best"
        os.makedirs(best_output_path, exist_ok=True)
        with open(os.path.join(best_output_path, "best_results.json"), "w") as f:
            json.dump(best_results, f, indent=2, ensure_ascii=False)
        if ckpt_num != 0:
            for dataset in datasets:
                best_ckpt_path = f"outputs/evaluation_{VERSION}/{curr_run_name}/{best_ckpt}/{dataset}.json"
                # copy the best ckpt result to the best path
                best_result = json.load(open(best_ckpt_path))
                with open(os.path.join(best_output_path, f"{dataset}.json"), "w") as f:
                    json.dump(best_result, f, indent=2, ensure_ascii=False)

        run.finish()
        time.sleep(10)


def main():
    """Main function to run all evaluations"""
    print("Starting comprehensive evaluation of all models on all datasets")
    print(f"Dirs: {', '.join(dirs)}")
    print(f"Datasets: {', '.join(datasets)}")

    run_all_evaluations()


if __name__ == "__main__":
    while True:
        main()
        # time.sleep(1800)
        break
