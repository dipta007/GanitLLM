import random
from collections import defaultdict

from datasets import Dataset, DatasetDict, load_dataset


def curriculum_shuffle(dataset):
    problems_by_difficulty = defaultdict(list)

    for problem in dataset:
        problems_by_difficulty[problem["correct_counts"]].append(problem)

    difficulties = list(problems_by_difficulty.keys())
    num_difficulties = len(difficulties)
    problems_per_difficulty = len(problems_by_difficulty[difficulties[0]])
    print("difficulties: ", difficulties)
    print("num_difficulties: ", num_difficulties)
    print("problems_per_difficulty: ", problems_per_difficulty)

    primary_per_block = int(problems_per_difficulty * 0.8)
    remaining_per_difficulty = problems_per_difficulty - primary_per_block
    remaining_per_difficulty = 31 * 3
    primary_per_block = problems_per_difficulty - remaining_per_difficulty
    print("primary_per_block: ", primary_per_block)
    print("remaining_per_difficulty: ", remaining_per_difficulty)
    base_mix_per_other = remaining_per_difficulty // (num_difficulties - 1)
    remainder = remaining_per_difficulty % (num_difficulties - 1)
    print("base_mix_per_other: ", base_mix_per_other)
    print("remainder: ", remainder)

    for difficulty in range(32, 0, -1):
        print(
            f"difficulty: {difficulty}, problems: {len(problems_by_difficulty[difficulty])}"
        )
        random.shuffle(problems_by_difficulty[difficulty])

    result = []

    for block_idx, primary_difficulty in enumerate(range(32, 0, -1)):
        block = []

        block.extend(problems_by_difficulty[primary_difficulty][:primary_per_block])

        for other_idx, other_difficulty in enumerate(range(32, 0, -1)):
            if other_difficulty != primary_difficulty:
                slice_idx = block_idx if other_idx > block_idx else block_idx - 1
                count = base_mix_per_other + (1 if slice_idx < remainder else 0)
                start_idx = (
                    primary_per_block
                    + slice_idx * base_mix_per_other
                    + min(slice_idx, remainder)
                )
                end_idx = start_idx + count
                print(
                    f"primary_difficulty: {primary_difficulty}, other_difficulty: {other_difficulty}, start_idx: {start_idx}, end_idx: {end_idx}"
                )

                block.extend(
                    problems_by_difficulty[other_difficulty][
                        start_idx : start_idx + count
                    ]
                )

        random.shuffle(block)
        result.extend(block)

    return result


def deduplicate_based_on_id(ds):
    taken_ids = set()
    train_ds = []
    for row in ds:
        if row["id"] in taken_ids:
            continue
        taken_ids.add(row["id"])
        train_ds.append(row)
    return Dataset.from_list(train_ds)


def print_dataset(ds):
    print(ds)
    print("========================================")
    print("size of ordered_train: ", len(ds["ordered_train"]))
    print("size of unordered_train: ", len(ds["unordered_train"]))
    print("size of dev: ", len(ds["dev"]))
    print("========================================")


if __name__ == "__main__":
    ds = load_dataset("GanitLLM/Ganit_balanced")
    train_ds = curriculum_shuffle(ds["train"])
    train_ds = deduplicate_based_on_id(train_ds)

    ds = DatasetDict(
        {"ordered_train": train_ds, "unordered_train": ds["train"], "dev": ds["dev"]}
    )
    for split in ["ordered_train", "unordered_train", "dev"]:
        print(f"Pushing {split} to hub")
        ds[split].push_to_hub("GanitLLM/Ganit_curriculum_shuffle", split, private=True)

    print_dataset(ds)
