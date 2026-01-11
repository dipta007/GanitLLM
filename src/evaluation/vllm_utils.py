import hashlib
import os
import random
import signal
import subprocess
import time
import uuid

import torch
from openai import OpenAI

from src.evaluation.eval_types import SamplerResponse

PORT = random.randint(8888, 9999)


def hash_str(s):
    hash_obj = hashlib.sha256(s.encode())
    return hash_obj.hexdigest()


def run_vllm(model_id, port=PORT, lora_path=None):
    assert model_id is not None, "Model ID is required"
    call_model_id = model_id
    if lora_path is not None:
        call_model_id = f"lora-{hash_str(lora_path)}"
    if is_vllm_ready(port, call_model_id):
        print(f"VLLM for {call_model_id} is already running on port {port}")
        return None, call_model_id

    device_count = torch.cuda.device_count()
    devices = ",".join(str(i) for i in range(device_count))
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    print(f"Running VLLM on {devices} for {call_model_id} on port {port}")
    print(f"CUDA_VISIBLE_DEVICES: {devices}")
    max_num_seqs = 256
    if any(arg in model_id.lower() for arg in ["70", "72", "120", "235"]):
        max_num_seqs = 64
    elif any(arg in model_id.lower() for arg in ["9", "14", "32", "30", "20"]):
        max_num_seqs = 128

    max_model_len = 4096
    num_of_devices = len(devices.split(","))
    if any(arg in model_id.lower() for arg in ["0.5", "0.6"]):
        num_of_devices = min(2, num_of_devices)

    cmd = f"""
    export CUDA_VISIBLE_DEVICES={devices} && \
        vllm serve {model_id} \
            --load-format safetensors \
            --gpu-memory-utilization 0.95 \
            --max-model-len {max_model_len} \
            --max-num-seqs {max_num_seqs} \
            --port {port} \
            --enable-prefix-caching \
            --block-size 16"""

    if "tigerllm-9b" in model_id.lower() or "tiger-9b" in model_id.lower():
        cmd += " --tokenizer md-nishat-008/TigerLLM-9B-it"
    elif "tigerllm-1b" in model_id.lower() or "tiger-1b" in model_id.lower():
        cmd += " --tokenizer md-nishat-008/TigerLLM-1B-it"

    if lora_path is not None:
        cmd += f" --enable-lora --lora-modules {call_model_id}={lora_path} --max-lora-rank 256"

    if lora_path is not None or any(arg in model_id.lower() for arg in ["32", "70"]):
        cmd += f" -tp {num_of_devices}"
    else:
        cmd += f" -dp {num_of_devices}"

    if "qwen3" in model_id.lower() and "instruct" not in model_id.lower():
        cmd += " --reasoning-parser qwen3"

    print("--------------------------------")
    print(cmd)
    print("--------------------------------")
    proc = subprocess.Popen(
        cmd,
        shell=True,
        preexec_fn=os.setsid,
    )

    wait_for_vllm([port], [call_model_id])
    return proc, call_model_id


def is_vllm_ready(port, model):
    client = OpenAI(api_key="empty", base_url=f"http://localhost:{port}/v1")
    try:
        models = client.models.list()
        if any(model == m.id for m in models.data):
            return True
        else:
            raise ValueError(f"VLLM for {model} is not ready")
    except:
        return False


def wait_for_vllm(ports, models):
    for port, model in zip(ports, models):
        while not is_vllm_ready(port, model):
            print(f"VLLM for {model} is not ready, waiting for 10 seconds...")
            time.sleep(10)


def kill_vllm(procs):
    for proc in procs:
        if proc is not None:
            os.killpg(proc.pid, signal.SIGTERM)

    # if any vllm was running, wait for it to be killed
    if any(proc is not None for proc in procs):
        print("Waiting for VLLM to be killed...")
        time.sleep(30)


def sampler(prompt_messages: list[dict[str, str]], port, model_id) -> SamplerResponse:
    # client = OpenAI(api_key="empty", base_url=f"http://localhost:{port}/v1")
    client = OpenAI()
    config = dict(
        model=model_id,
        messages=prompt_messages,
        timeout=1000,
        temperature=0.0,
    )
    if prompt_messages[0]["role"] == "assistant":
        config["add_generation_prompt"] = False
        config["continue_final_message"] = True

    try:
        response = client.chat.completions.create(**config)
        response_text = response.choices[0].message.content
        if (
            "reasoning_content" in response.choices[0].message
            and response.choices[0].message.reasoning_content is not None
        ):
            reasoning_content = response.choices[0].message.reasoning_content
            response_text = f"<think>{reasoning_content}</think>\n{response_text}"
    except Exception as e:
        print(f"&&&& Error: {e}")
        print(f"&&&& Config: {config}")
        raise e

    return SamplerResponse(
        response_text=response_text,
        actual_queried_message_list=prompt_messages,
        finish_reason=response.choices[0].finish_reason,
    )
