import random
from collections import Counter

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm

from src.prompt import SYSTEM_PROMPT

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def print_historical_datasets():
    ds1 = load_dataset("GanitLLM/GANIT-v2")
    ds2 = load_dataset("GanitLLM/GANIT_filtered_v2")
    ds3 = load_dataset("GanitLLM/GANIT_filtered_fuzz_v2")
    ds4 = load_dataset("GanitLLM/GANIT_filtered_fuzz_minhash_v2")
    ds5 = load_dataset("GanitLLM/GANIT_filtered_fuzz_minhash_decontamination_v2")

    print("==>> Running print_historical_datasets")
    print("========================================")
    print("1. GanitLLM/GANIT-v2")
    print(ds1)
    print("2. GanitLLM/GANIT_filtered_v2")
    print(ds2)
    print("3. GanitLLM/GANIT_filtered_fuzz_v2")
    print(ds3)
    print("4. GanitLLM/GANIT_filtered_fuzz_minhash_v2")
    print(ds4)
    print("5. GanitLLM/GANIT_filtered_fuzz_minhash_decontamination_v2")
    print(ds5)

    counters = Counter(ds5["train"]["source_name"])
    print("source counts: ", counters)
    print("========================================")


def get_combined_dataset(push_to_hub=False):
    print("==>> Running get_combined_dataset")
    dss = []
    print("========================================")
    for i in range(1, 5):
        try:
            ds = load_dataset(f"GanitLLM/tmp_{i}_tagged_difficulty")
            dss.append(ds["train"])
            print(f"Loaded dataset {i}")
        except Exception as e:
            print(f"Error loading dataset {i}: {e}")

    # Combine
    cds = concatenate_datasets(dss)
    cds = cds.rename_column("difficulties", "difficulty")
    if push_to_hub:
        cds.push_to_hub("GanitLLM/Ganit_imbalanced")
        print("Combined dataset pushed to hub")
        print(cds)

    for i in range(1, 5):
        print(f"size of tmp_{i}_tagged_difficulty: {len(dss[i - 1])}")
    print(f"size of Ganit_imbalanced: {len(cds)}")

    difficulty_counts = Counter(cds["difficulty"])
    print("difficulty counts: ")
    for c in ["easy", "medium", "hard", "olympiad"]:
        print(f"{c}: {difficulty_counts.get(c, 0)}")

    correct_counter = Counter(cds["correct_counts"])
    print("correct counts: ")
    for i in range(33):
        print(f"{i}: {correct_counter.get(i, 0)}")
    print("========================================")

    return cds


def filter_unsolved_dataset(ds):
    print("==>> Running filter_unsolved_dataset")
    # remove the onew where correct_counts is 0
    nds = ds.filter(lambda x: x["correct_counts"] > 0)
    print("========================================")
    print("Before removing the ones where correct_counts is 0: ", len(ds))
    print("After removing the ones where correct_counts is 0: ", len(nds))
    correct_counter = Counter(nds["correct_counts"])
    for i in range(33):
        print(f"{i}: {correct_counter.get(i, 0)}")
    print("========================================")
    return nds


def refined_difficulty(ds):
    print("==>> Running refined_difficulty")

    def get_refined_difficulty(row):
        # bucket of 8 for correct_counts
        if row["correct_counts"] <= 8:
            return "olympiad"
        elif row["correct_counts"] <= 16:
            return "hard"
        elif row["correct_counts"] <= 24:
            return "medium"
        else:
            return "easy"

    refined_difficulty = []
    for row in ds:
        refined_difficulty.append(get_refined_difficulty(row))
    ds = ds.add_column("refined_difficulty", refined_difficulty)

    print("========================================")
    print("Change on difficulty: ")
    for c in ["easy", "medium", "hard", "olympiad"]:
        print(
            f"{c}: {Counter(ds['difficulty']).get(c, 0)} -> {Counter(ds['refined_difficulty']).get(c, 0)}"
        )
    print("========================================")
    return ds


def split_dataset_by_correct_counts(ds, dev_size=50):
    print("==>> Running split_dataset_by_correct_counts")
    train_set = []
    dev_set = []

    counter = {i: dev_size for i in range(1, 33)}

    # shuffle the dataset
    ds = ds.shuffle(seed=RANDOM_SEED)

    # take dev_size examples for each difficulty
    for row in tqdm(ds, desc="Splitting dataset by correct counts"):
        correct_counts = row["correct_counts"]
        if counter[correct_counts] > 0:
            dev_set.append(row)
            counter[correct_counts] -= 1
        else:
            train_set.append(row)

    train_ds = Dataset.from_list(train_set)
    dev_ds = Dataset.from_list(dev_set)
    pds = DatasetDict({"train": train_ds, "dev": dev_ds})

    print("========================================")
    print("Non balanced train set: ", len(pds["train"]))
    print("Non balanced dev set: ", len(pds["dev"]))
    print(pds)
    print("========================================")

    return pds


def get_balanced_dataset_by_correct_counts(ds, push_to_hub=False):
    print("==>> Running get_balanced_dataset_by_correct_counts")
    # shuffle the dataset
    ds = ds.shuffle(seed=RANDOM_SEED)

    counter = Counter(ds["train"]["correct_counts"])
    min_count = min(counter.values())
    train_counter = {i: min_count for i in range(1, 33)}
    print("========================================")
    print("expected train set size: ", train_counter)

    train_set = []
    for row in tqdm(ds["train"], desc="Balancing dataset by correct counts"):
        correct_counts = row["correct_counts"]
        if train_counter[correct_counts] > 0:
            train_set.append(row)
            train_counter[correct_counts] -= 1

    train_ds = Dataset.from_list(train_set)
    dev_ds = ds["dev"]
    # remove difficulties column
    train_ds = train_ds.remove_columns(["difficulty"])
    dev_ds = dev_ds.remove_columns(["difficulty"])
    train_ds = train_ds.rename_column("refined_difficulty", "difficulty")
    dev_ds = dev_ds.rename_column("refined_difficulty", "difficulty")

    pds = DatasetDict({"train": train_ds, "dev": dev_ds})

    print("Balanced train set: ", len(pds["train"]))
    print("Balanced dev set: ", len(pds["dev"]))
    print(pds)
    if push_to_hub:
        pds.push_to_hub("GanitLLM/Ganit_balanced", private=True)
        print("Balanced dataset pushed to hub")
        print(pds)

    print("========================================")

    return pds


def get_gpt_formatted_dataset(ds):
    print("==>> Running get_gpt_formatted_dataset")

    def format_to_messages(row):
        messages = [
            {"role": "user", "content": SYSTEM_PROMPT.format(problem=row["problem"])},
        ]
        row["messages"] = messages
        return row

    ds = ds.remove_columns(["metadata", "generated_solutions", "prompts"])
    return ds.map(format_to_messages)


def stat_final_dataset(ds):
    print("==>> Running stat_final_dataset")
    for split in ["train", "dev"]:
        print("========================================")
        print(f"Split: {split}")
        source_counts = Counter(ds[split]["source_name"])
        print("source counts: ")
        for c in source_counts:
            print(f"{c}: {source_counts.get(c, 0)}")

        difficulty_counts = Counter(ds[split]["difficulty"])
        print("difficulty counts: ")
        for c in ["easy", "medium", "hard", "olympiad"]:
            print(f"{c}: {difficulty_counts.get(c, 0)}")

        correct_counts_counts = Counter(ds[split]["correct_counts"])
        print("correct counts:")
        for i in range(33):
            print(f"{i}: {correct_counts_counts.get(i, 0)}")
        print("========================================")

    print("========================================")
    print("Final dataset: ", len(ds[split]))
    print(ds)
    print("========================================")


if __name__ == "__main__":
    print_historical_datasets()
    PUSH_TO_HUB = True

    ds = get_combined_dataset(push_to_hub=PUSH_TO_HUB)
    ds = filter_unsolved_dataset(ds)
    ds = refined_difficulty(ds)
    ds = split_dataset_by_correct_counts(ds, dev_size=30)
    ds = get_gpt_formatted_dataset(ds)
    ds = get_balanced_dataset_by_correct_counts(ds, push_to_hub=PUSH_TO_HUB)

    stat_final_dataset(ds)
