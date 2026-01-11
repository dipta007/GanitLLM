# GanitLLM: Bengali Mathematical Reasoning with Language Models

[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/dipta007/Ganit)
[![Models](https://img.shields.io/badge/HuggingFace-Models-orange)](https://huggingface.co/collections/dipta007/ganitllm)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

Official implementation of **"GanitLLM: Bengali Mathematical Reasoning with Language Models"**

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Models](#models)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)

## Overview

GanitLLM is the first comprehensive framework for training and evaluating Bengali mathematical reasoning capabilities in language models. We introduce:

- **Ganit Dataset**: A curated dataset of 25k Bengali math problems with difficulty-stratified splits
- **Curriculum GRPO (CGRPO)**: A novel training approach combining curriculum learning with Group Relative Policy Optimization
- **GanitLLM Models**: A family of models (0.6B, 1.7B, 4B parameters) achieving state-of-the-art performance on Bengali math reasoning

## Key Features

- Multi-stage training pipeline: SFT → Curriculum Learning → GRPO
- Difficulty-aware data curation using ensemble LLM evaluation
- Support for both "thinking" and "instruct" model variants
- Comprehensive evaluation on MGSM, MSVAMP, and in-domain test sets
- Efficient training with LoRA, FlashAttention, and vLLM

## Installation

### Requirements
- Python 3.12+
- CUDA 12.8+
- GPU with at least 24GB VRAM (for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/dipta007/GanitLLM.git
cd GanitLLM

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
make setup

# Or manually:
uv venv --python 3.12 --seed
source .venv/bin/activate
uv sync
```

### Flash Attention (Optional, recommended)
```bash
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation --no-cache
```

## Dataset

The Ganit dataset is available on HuggingFace:

| Name | Description | Size |
|------|-------------|------|
| [dipta007/Ganit](https://huggingface.co/datasets/dipta007/Ganit) | GanitSFT + GanitRLVR + GanitDEV | 25k rows |

### Dataset Structure

```python
from datasets import load_dataset

dataset = load_dataset("dipta007/Ganit")
```

Each example contains:
- `problem`: Bengali math problem statement
- `bengali_solution`: Answer in Bengali numerals
- `english_solution`: Answer in English numerals
- `difficulty`: One of {easy, medium, hard, olympiad}
- `messages`: Chat-formatted training data

### Difficulty Distribution

| Difficulty | Criteria | Count |
|------------|----------|-------|
| Easy | >75% LLMs correct | ~4k |
| Medium | 50-75% LLMs correct | ~4k |
| Hard | 25-50% LLMs correct | ~4k |
| Olympiad | <25% LLMs correct | ~4k |

## Models

### Pre-trained Models

All models are available on HuggingFace:

| Model | Parameters | Training | HuggingFace Link |
|-------|------------|----------|------------------|
| GanitLLM-4B_SFT_CGRPO | 4B | SFT + CGRPO | [Link](https://huggingface.co/dipta007/GanitLLM-4B_SFT_CGRPO) |
| GanitLLM-4B_SFT_GRPO | 4B | SFT + GRPO | [Link](https://huggingface.co/dipta007/GanitLLM-4B_SFT_GRPO) |
| GanitLLM-4B_SFT | 4B | SFT | [Link](https://huggingface.co/dipta007/GanitLLM-4B_SFT) |
| GanitLLM-4B_CGRPO | 4B | CGRPO | [Link](https://huggingface.co/dipta007/GanitLLM-4B_CGRPO) |
| GanitLLM-1.7B_SFT_CGRPO | 1.7B | SFT + CGRPO | [Link](https://huggingface.co/dipta007/GanitLLM-1.7B_SFT_CGRPO) |
| GanitLLM-1.7B_SFT_GRPO | 1.7B | SFT + GRPO | [Link](https://huggingface.co/dipta007/GanitLLM-1.7B_SFT_GRPO) |
| GanitLLM-1.7B_SFT | 1.7B | SFT | [Link](https://huggingface.co/dipta007/GanitLLM-1.7B_SFT) |
| GanitLLM-1.7B_CGRPO | 1.7B | CGRPO | [Link](https://huggingface.co/dipta007/GanitLLM-1.7B_CGRPO) |
| GanitLLM-0.6B_SFT_CGRPO | 0.6B | SFT + CGRPO | [Link](https://huggingface.co/dipta007/GanitLLM-0.6B_SFT_CGRPO) |
| GanitLLM-0.6B_SFT_GRPO | 0.6B | SFT + GRPO | [Link](https://huggingface.co/dipta007/GanitLLM-0.6B_SFT_GRPO) |
| GanitLLM-0.6B_SFT | 0.6B | SFT | [Link](https://huggingface.co/dipta007/GanitLLM-0.6B_SFT) |
| GanitLLM-0.6B_CGRPO | 0.6B | CGRPO | [Link](https://huggingface.co/dipta007/GanitLLM-0.6B_CGRPO) |

**Model Types:**
- **SFT_CGRPO**: Best performing models with multi-stage training (SFT → Curriculum GRPO)
- **SFT_GRPO**: Models with standard SFT followed by GRPO
- **SFT**: Models with supervised fine-tuning only (foundation for RL training)
- **CGRPO**: Models trained with curriculum GRPO only (no SFT)

## Usage

### Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "dipta007/GanitLLM-4B_SFT_CGRPO"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Format the prompt
problem = "একটি দোকানে ১২টি আপেল আছে। যদি ৫টি আপেল বিক্রি হয়, তাহলে কতটি আপেল বাকি থাকবে?"

prompt = f"""A conversation takes place between the user and the assistant. The user asks questions, and the assistant solves them. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Reasoning and answer must be in Bengali. Final answer must be a number nothing else.

Question: {problem}"""

messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)

outputs = model.generate(inputs.to(model.device), max_new_tokens=2048, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```



### Training

#### Supervised Fine-Tuning (SFT)

```bash
PYTHONPATH=. uv run src/training/sft.py \
    --model_name Qwen/Qwen3-4B \
    --dataset_path data/sft/train.jsonl \
    --output_dir checkpoints/sft
```

#### GRPO Training

```bash
PYTHONPATH=. uv run src/training/unsloth/grpo_lora.py \
    --model_name checkpoints/sft \
    --dataset_path data/rl/train.jsonl \
    --output_dir checkpoints/grpo 
```

#### Curriculum GRPO (CGRPO)

First, prepare curriculum-ordered data:
```bash
PYTHONPATH=. uv run src/data_gen/curriculum_shuffle.py \
    --input data/rl/train.jsonl \
    --output data/rl/train_curriculum.jsonl
```

Then run GRPO with curriculum data:
```bash
PYTHONPATH=. uv run src/training/unsloth/grpo_lora.py \
    --model_name checkpoints/sft \
    --dataset_path data/rl/train_curriculum.jsonl \
    --output_dir checkpoints/cgrpo
```

### Evaluation

Evaluate on MGSM Bengali:
```bash
PYTHONPATH=. uv run src/evaluation/eval.py \
    --model dipta007/GanitLLM-4B_SFT_CGRPO \
    --dataset mgsm \
    --output_dir results/
```

## Data Processing Pipeline

To process your own Bengali math data:

```bash
# Step 1: Filter and validate Bengali content
PYTHONPATH=. uv run src/data_gen/0.filter.py --input raw_data.jsonl --output filtered.jsonl

# Step 2: Remove exact duplicates
PYTHONPATH=. uv run src/data_gen/1.deduplication.py --input filtered.jsonl --output deduped.jsonl

# Step 3: Remove near-duplicates using MinHash
PYTHONPATH=. uv run src/data_gen/2.minhash.py --input deduped.jsonl --output minhash.jsonl

# Step 4: Decontaminate against test sets
PYTHONPATH=. uv run src/data_gen/3.decontamination.py --input minhash.jsonl --output clean.jsonl

# Step 5: Tag difficulty using ensemble LLM evaluation
PYTHONPATH=. uv run src/data_gen/4.tag_difficulty.py --input clean.jsonl --output final.jsonl
```

Or run the full pipeline:
```bash
make process-dataset INPUT=raw_data.jsonl OUTPUT=final.jsonl
```


## Reward Functions

Our GRPO training uses multi-component rewards:

1. **Format Reward**: Validates `<think>` and `<answer>` tag structure
2. **Bengali Reasoning Reward**: Ensures >80% Bengali text in reasoning
3. **Answer Correctness**:
   - Bengali numeral match: +2.0
   - English numeral match: +1.0
4. **Length Penalty**: Soft penalty for responses >2000 tokens

## Results

For results and more details, please refer to our [paper](https://arxiv.org/).

## Citation

If you find this work useful, please cite our paper:

```bibtex
will be updated
```

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [vLLM](https://github.com/vllm-project/vllm) for fast inference
- [Qwen](https://github.com/QwenLM/Qwen) for base models

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
