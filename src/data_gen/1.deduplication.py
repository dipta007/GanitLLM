import time
import unicodedata
from collections import defaultdict

from datasets import load_dataset
from rapidfuzz import fuzz
from tqdm import tqdm


def deduplicate_strings(strings, ids, threshold=70, ngram_size=3):
    """
    The ngram_size parameter can be tuned
    - smaller values (2-3) catch more similarities but do more comparisons
    - larger values (4-5) are faster but might miss some duplicates.
    """

    def normalize_text(text):
        text = unicodedata.normalize("NFKC", text)
        text = " ".join(text.split())
        return text.strip().lower()

    def get_ngrams(text, n):
        return set(text[i : i + n] for i in range(len(text) - n + 1))

    unique_strings = []
    unique_ids = []
    normalized_unique = []
    ngram_buckets = defaultdict(set)

    for idx, string in tqdm(
        enumerate(strings), total=len(strings), desc="Deduplicating strings"
    ):
        normalized = normalize_text(string)

        if len(normalized) < ngram_size:
            candidate_indices = set(range(len(normalized_unique)))
        else:
            string_ngrams = get_ngrams(normalized, ngram_size)
            candidate_indices = set()

            for ngram in string_ngrams:
                candidate_indices.update(ngram_buckets[ngram])

        is_duplicate = False
        for candidate_idx in candidate_indices:
            similarity = fuzz.ratio(normalized, normalized_unique[candidate_idx])
            if similarity >= threshold:
                is_duplicate = True
                # print(f"First string: {string}")
                # print(f"Second string: {normalized_unique[candidate_idx]}")
                # print(f"Similarity: {similarity}")
                # print("--------------------------------")
                break

        if not is_duplicate:
            current_idx = len(unique_strings)
            unique_strings.append(string)
            unique_ids.append(ids[idx])
            normalized_unique.append(normalized)

            if len(normalized) >= ngram_size:
                string_ngrams = get_ngrams(normalized, ngram_size)
                for ngram in string_ngrams:
                    ngram_buckets[ngram].add(current_idx)

    return unique_strings, unique_ids


if __name__ == "__main__":
    ds = load_dataset("GanitLLM/GANIT_filtered_v2")
    print(ds)

    strings = ds["train"]["problem"]
    ids = ds["train"]["id"]
    print(len(strings))

    start_time = time.time()
    deduplicated_strings, deduplicated_ids = deduplicate_strings(strings, ids)
    print(len(deduplicated_strings))
    print(len(deduplicated_ids))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Percentage of duplicates: {len(deduplicated_ids) / len(ids) * 100:.2f}%")
    print(f"Size improvement from {len(ids)} to {len(deduplicated_ids)}")

    deduplicated_ids = set(deduplicated_ids)
    ds["train"] = ds["train"].filter(lambda x: x["id"] in deduplicated_ids)
    print(ds)
    ds.push_to_hub("GanitLLM/GANIT_filtered_fuzz_v2", private=True)
