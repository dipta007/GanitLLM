import hashlib
import time
from typing import List, Set
from datasets import load_dataset
from tqdm import tqdm


def deduplicate_strings_minhash(
    strings_A: List[str], strings_B: List[str], ids_B: List[int]
):
    if not strings_B:
        return [], []

    num_hashes = 200
    threshold = 0.5

    def get_shingles(text: str, k: int = 3) -> Set[str]:
        if len(text) < k:
            return {text}
        return {text[i : i + k] for i in range(len(text) - k + 1)}

    def hash_func(data: str, seed: int) -> int:
        return int(hashlib.md5(f"{seed}{data}".encode("utf-8")).hexdigest(), 16)

    def minhash_signature(shingles: Set[str], num_hashes: int) -> List[int]:
        signature = []
        for i in range(num_hashes):
            min_hash = (
                min(hash_func(shingle, i) for shingle in shingles) if shingles else 0
            )
            signature.append(min_hash)
        return signature

    def jaccard_similarity(sig1: List[int], sig2: List[int]) -> float:
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    signatures_A = []
    for string in tqdm(strings_A, total=len(strings_A), desc="Minhashing strings A"):
        shingles = get_shingles(string)
        signature = minhash_signature(shingles, num_hashes)
        signatures_A.append(signature)

    signatures_B = []
    for string in tqdm(strings_B, total=len(strings_B), desc="Minhashing strings B"):
        shingles = get_shingles(string)
        signature = minhash_signature(shingles, num_hashes)
        signatures_B.append(signature)

    to_remove = set()
    for i in tqdm(
        range(len(strings_B)), total=len(strings_B), desc="Finding duplicates"
    ):
        if i in to_remove:
            continue
        for j in range(len(strings_A)):
            similarity = jaccard_similarity(signatures_B[i], signatures_A[j])
            if similarity >= threshold:
                to_remove.add(i)
                print(f"First string: {strings_B[i]}")
                print(f"Second string: {strings_A[j]}")
                print(f"Similarity: {similarity}")
                print("--------------------------------")
                break

    return [strings_B[i] for i in range(len(strings_B)) if i not in to_remove], [
        ids_B[i] for i in range(len(strings_B)) if i not in to_remove
    ]


if __name__ == "__main__":
    # test dataset
    DATASET_TO_HF = {
        "mgsm": "jbross-ibm-research/mgsm",
        "msvamp": "Mathoctopus/MSVAMP",
    }
    ds_A_1 = load_dataset(DATASET_TO_HF["mgsm"], "bn", split="test")
    ds_A_2 = load_dataset(DATASET_TO_HF["msvamp"], "bn", split="test")
    str_a_1 = list(ds_A_1["question"])
    str_a_2 = list(ds_A_2["m_query"])
    str_a = str_a_1 + str_a_2

    # real dataset
    ds_B = load_dataset("GanitLLM/GANIT_filtered_fuzz_minhash_v2")
    str_b = list(ds_B["train"]["problem"])
    ids_b = list(ds_B["train"]["id"])

    print("Length of strings_A", len(str_a))
    print("Length of strings_B", len(str_b))

    start_time = time.time()
    deduplicated_strings, deduplicated_ids = deduplicate_strings_minhash(
        str_a, str_b, ids_b
    )
    print(len(deduplicated_strings))
    print(len(deduplicated_ids))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Percentage of duplicates: {len(deduplicated_ids) / len(ids_b) * 100:.2f}%")
    print(f"Size improvement from {len(ids_b)} to {len(deduplicated_ids)}")

    deduplicated_ids = set(deduplicated_ids)
    ds_B["train"] = ds_B["train"].filter(lambda x: x["id"] in deduplicated_ids)
    print(ds_B)
    ds_B.push_to_hub(
        "GanitLLM/GANIT_filtered_fuzz_minhash_decontamination_v2", private=True
    )
