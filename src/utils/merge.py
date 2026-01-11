import os
from unsloth import FastModel


def merge_model(lora_path: str, tmp_dir: str, max_seq_length: int = 4096) -> str:
    model, tokenizer = FastModel.from_pretrained(
        model_name=lora_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # Must be False for 16bit merge
    )
    os.makedirs("./tmp", exist_ok=True)
    # model.save_pretrained_merged("./tmp", tokenizer, save_method="merged_16bit")
    model.push_to_hub_merged(
        tmp_dir, tokenizer, save_method="merged_16bit", private=True
    )
    return tmp_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--tmp_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    args = parser.parse_args()
    merge_model(args.lora_path, args.tmp_dir, args.max_seq_length)
