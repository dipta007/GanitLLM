import json
import os

# os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
# os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

import unsloth
import jsonlines
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from vllm import SamplingParams

from src.training.unsloth.reward_functions import (
    response_one_reward_rules_all,
    soft_overlong_reward,
)

VERSION = "v6"
RANDOM_SEED = 42
NPROC_PER_NODE = torch.cuda.device_count()
DEVICE_IDS = list(range(NPROC_PER_NODE))

GLOBAL_BATCH_SIZE = 32
BATCH_SIZE = 8
GRAD_ACC_STEPS = GLOBAL_BATCH_SIZE / (BATCH_SIZE * NPROC_PER_NODE)
if GLOBAL_BATCH_SIZE % (BATCH_SIZE * NPROC_PER_NODE) != 0:
    raise ValueError(
        f"Global batch size {GLOBAL_BATCH_SIZE} is not divisible by {BATCH_SIZE * NPROC_PER_NODE}"
    )
GRAD_ACC_STEPS = int(GRAD_ACC_STEPS)


def get_dataset(args):
    dataset_name = ""
    if args.thinking:
        dataset_name += "thinking"
    else:
        dataset_name += "instruct"
    if args.ordered:
        dataset_name += "_ordered"
    else:
        dataset_name += "_unordered"
    with jsonlines.open(f"{args.dataset_path}/{dataset_name}_train.jsonl") as reader:
        dataset = []
        for line in reader:
            dataset.append(line)
        dataset_ds = Dataset.from_list(dataset)
        dataset_ds = dataset_ds.add_column("prompt", dataset_ds["messages"])
        dataset_ds = dataset_ds.remove_columns(["messages"])
    return dataset_ds


def main(args):
    output_path = f"/umbc/rs/pi_ferraro/users/sroydip1/GanitLLM_checkpoints/grpo_{VERSION}/{args.run_name}"

    dataset_ds = get_dataset(args)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_8bit=False,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=0.8,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_SEED,
    )

    vllm_sampling_params = SamplingParams(
        seed=RANDOM_SEED,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        learning_rate=1e-5,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1e-6},
        warmup_ratio=0,
        max_grad_norm=1.0,
        # optim="adamw_8bit",
        optim="adamw_torch",
        weight_decay=0.001,
        logging_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        num_generations=args.num_generations,
        max_prompt_length=1000,
        max_completion_length=3000,
        num_train_epochs=5,
        save_steps=50,
        report_to="wandb",
        log_completions=True,
        num_completions_to_print=0,
        output_dir=output_path,
        epsilon=0.2,
        epsilon_high=0.28,
        loss_type="bnpo",
        beta=0.1,
        mask_truncated_completions=True,
        shuffle_dataset=(not args.ordered),
        ddp_find_unused_parameters=False,
        # For optional training + evaluation
        # fp16_full_eval = True,
        # per_device_eval_batch_size = 4,
        # eval_accumulation_steps = 1,
        # eval_strategy = "steps",
        # eval_steps = 1,
    )

    # For optional training + evaluation
    # new_dataset = dataset.train_test_split(test_size = 0.01)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            response_one_reward_rules_all,
            soft_overlong_reward,
        ],
        args=training_args,
        train_dataset=dataset_ds,
        # For optional training + evaluation
        # train_dataset = new_dataset["train"],
        # eval_dataset = new_dataset["test"],
    )

    resume_from_checkpoint = False
    if (
        os.path.exists(output_path)
        and len([f for f in os.listdir(output_path) if f.startswith("checkpoint-")]) > 0
    ):
        resume_from_checkpoint = True
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def print_hyperparameters(args):
    print("========================================")
    print("Global Variables:")
    print("========================================")
    print(f"DEVICE_IDS: {DEVICE_IDS}")
    print(f"GLOBAL_BATCH_SIZE: {GLOBAL_BATCH_SIZE}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"GRAD_ACC_STEPS: {GRAD_ACC_STEPS}")
    print(f"RANDOM_SEED: {RANDOM_SEED}")
    print("========================================")
    print("Arguments:")
    print("========================================")
    print(json.dumps(args.__dict__, indent=4))
    print("========================================")
    print("WANDB Variables:")
    print("========================================")
    for key, value in os.environ.items():
        if key.startswith("WANDB_"):
            print(f"{key}: {value}")
    print("========================================")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", "-d", type=str, default="data_v2/rl")
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--model_name", "-m", type=str, required=True)
    parser.add_argument("--run_name", "-r", type=str, required=True)
    parser.add_argument("--ordered", "-o", action="store_true", default=False)
    parser.add_argument("--thinking", "-t", action="store_true", default=False)
    args = parser.parse_args()
    args.model_id = args.model_name.split("/")[-1]
    args.run_name = f"{args.run_name}_lora_b0_{args.model_id}_{VERSION}"

    # set wandb environment variables
    os.environ["WANDB_RUN_ID"] = args.run_name
    os.environ["WANDB_RESUME"] = "auto"
    os.environ["WANDB_ENTITY"] = "collab-srd"
    os.environ["WANDB_PROJECT"] = "GanitLLM-train"
    os.environ["WANDB_NAME"] = args.run_name
    os.environ["WANDB_TAGS"] = f"grpo_{VERSION},lora,unsloth,{args.model_id}"

    # run the training
    print_hyperparameters(args)
    main(args)

"""
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python src/training/unsloth/grpo_lora.py -m unsloth/Qwen3-0.6B -r cgrpo -o -t
CUDA_VISIBLE_DEVICES=1 BENGALI_REASONING_REWARD=True PYTHONPATH=. python src/training/unsloth/grpo_lora.py -m dipta007/GanitLLM-0.6B-SFT-1404 -r bn_cgrpo -o -t
CUDA_VISIBLE_DEVICES=2 BENGALI_REASONING_REWARD=True PYTHONPATH=. python src/training/unsloth/grpo_lora.py -m dipta007/GanitLLM-0.6B-SFT-1404 -r bn_grpo -t

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python src/training/unsloth/grpo_lora.py -m unsloth/Qwen3-1.7B -r cgrpo -o -t
CUDA_VISIBLE_DEVICES=1 BENGALI_REASONING_REWARD=True PYTHONPATH=. python src/training/unsloth/grpo_lora.py -m dipta007/GanitLLM-1.7B-SFT-1332 -r bn_cgrpo -o -t
CUDA_VISIBLE_DEVICES=2 BENGALI_REASONING_REWARD=True PYTHONPATH=. python src/training/unsloth/grpo_lora.py -m dipta007/GanitLLM-1.7B-SFT-1332 -r bn_grpo -t

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python src/training/unsloth/grpo_lora.py -m unsloth/Qwen3-4B -r cgrpo -o -t
CUDA_VISIBLE_DEVICES=1 BENGALI_REASONING_REWARD=True PYTHONPATH=. python src/training/unsloth/grpo_lora.py -m dipta007/GanitLLM-4B-SFT-432 -r bn_cgrpo -o -t
CUDA_VISIBLE_DEVICES=2 BENGALI_REASONING_REWARD=True PYTHONPATH=. python src/training/unsloth/grpo_lora.py -m dipta007/GanitLLM-4B-SFT-432 -r bn_grpo -t
"""
