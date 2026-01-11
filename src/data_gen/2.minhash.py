import hashlib
import time
from typing import List, Set
from datasets import load_dataset
from tqdm import tqdm


def deduplicate_strings_minhash(strings: List[str], ids: List[int]) -> List[str]:
    if not strings:
        return []

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

    signatures = []
    for string in tqdm(strings, total=len(strings), desc="Minhashing strings"):
        shingles = get_shingles(string)
        signature = minhash_signature(shingles, num_hashes)
        signatures.append(signature)

    duplicates = set()
    for i in tqdm(range(len(strings)), total=len(strings), desc="Finding duplicates"):
        if i in duplicates:
            continue
        for j in range(i + 1, len(strings)):
            if j in duplicates:
                continue
            similarity = jaccard_similarity(signatures[i], signatures[j])
            if similarity >= threshold:
                duplicates.add(j)
                # print(f"First string: {strings[i]}")
                # print(f"Second string: {strings[j]}")
                # print(f"Similarity: {similarity}")
                # print("--------------------------------")

    return [strings[i] for i in range(len(strings)) if i not in duplicates], [
        ids[i] for i in range(len(strings)) if i not in duplicates
    ]


if __name__ == "__main__":
    ds = load_dataset("GanitLLM/GANIT_filtered_fuzz_v2")
    print(ds)

    strings = ds["train"]["problem"]
    ids = ds["train"]["id"]
    print(len(strings))

    start_time = time.time()
    deduplicated_strings, deduplicated_ids = deduplicate_strings_minhash(strings, ids)
    print(len(deduplicated_strings))
    print(len(deduplicated_ids))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Percentage of duplicates: {len(deduplicated_ids) / len(ids) * 100:.2f}%")
    print(f"Size improvement from {len(ids)} to {len(deduplicated_ids)}")

    deduplicated_ids = set(deduplicated_ids)
    ds["train"] = ds["train"].filter(lambda x: x["id"] in deduplicated_ids)
    print(ds)
    ds.push_to_hub("GanitLLM/GANIT_filtered_fuzz_minhash_v2", private=True)
