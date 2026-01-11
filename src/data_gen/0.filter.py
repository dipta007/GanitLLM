import re
import string
import os
from tqdm import tqdm

from datasets import load_dataset

BENGALI_PERCENTAGE = 0.99


# def is_source_includes(row):
# return row["source_name"] == "BdMO"
# remove_sources = ["bangla-math-cot-dataset", "BMWP_Dataset"]
# return row["source_name"] not in remove_sources


def is_valid_row(row):
    if not isinstance(row["problem"], str):
        return False
    if not isinstance(row["solution"], str):
        return False
    if len(row["problem"]) == 0 or len(row["solution"]) == 0:
        return False
    return True


def is_multiple_choice(row):
    if "উত্তর পছন্দ: (A)" in row["problem"]:
        return False
    if (
        "A: " in row["problem"]
        and "B: " in row["problem"]
        and "C: " in row["problem"]
        and "D: " in row["problem"]
    ):
        return False
    return True


def is_bengali_in_question(row):
    def calculate_bengali_percentage(question):
        question = re.sub(r"\s+", "", question)
        bengali_chars = 0
        total_chars = len(question)
        for char in question:
            if (
                char
                in "অআইঈউঊঋঌএঐঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ঺঻়ঽািীুূৃৄেৈোৌ্ৎৗড়ঢ়য়ৠৡৢৣ০১২৩৪৫৬৭৮৯।ওংঃ"
                + string.punctuation
            ):
                bengali_chars += 1
            elif char in "0123456789":
                bengali_chars += 1
            else:
                pass
        return bengali_chars / total_chars

    bengali_percentage = calculate_bengali_percentage(row["problem"])
    return (
        bengali_percentage > BENGALI_PERCENTAGE
        or row["source_name"] == "dl_bdmo"
        or "s1k-32-Bangla" in row["source_name"]
    )


def is_solution_verifiable(row):
    solution = row["solution"]
    processed_solution = ""
    for char in solution:
        if char in "0123456789":
            processed_solution += char
        elif char in "০১২৩৪৫৬৭৮৯":
            mapping = {
                "০": "0",
                "১": "1",
                "২": "2",
                "৩": "3",
                "৪": "4",
                "৫": "5",
                "৬": "6",
                "৭": "7",
                "৮": "8",
                "৯": "9",
            }
            processed_solution += mapping[char]
        elif char in ".":
            processed_solution += "."
        elif char in ",":
            pass
        else:
            return False
    return processed_solution.strip() != ""


def filter_and_print_stat(ds, filter_func):
    total = len(ds)
    prev_counter = {}
    for row in tqdm(ds, total=len(ds)):
        prev_counter[row["source_name"]] = prev_counter.get(row["source_name"], 0) + 1
    ds = ds.filter(filter_func)
    counter = {}
    for row in tqdm(ds, total=len(ds)):
        counter[row["source_name"]] = counter.get(row["source_name"], 0) + 1
    # make a nice output with counter and filter_func.__name__
    print("-" * 100)
    print(f"After '{filter_func.__name__}' filtering: {total} -> {len(ds)}")
    for source_name, count in counter.items():
        print(f"{source_name}: {prev_counter[source_name]} -> {count}")
    print("-" * 100)
    return ds


def process_solution(row):
    def convert_solution_to_bengali(row):
        processed_solution = ""
        for char in row["solution"]:
            if char in "0123456789":
                mapping = {
                    "0": "০",
                    "1": "১",
                    "2": "২",
                    "3": "৩",
                    "4": "৪",
                    "5": "৫",
                    "6": "৬",
                    "7": "৭",
                    "8": "৮",
                    "9": "৯",
                }
                processed_solution += mapping[char]
            elif char in "০১২৩৪৫৬৭৮৯":
                processed_solution += char
            elif char in ".":
                processed_solution += "."
            elif char in ",":
                pass
            else:
                print(f"error: {char}")
                raise ValueError(f"error: {char}")
        row["bengali_solution"] = processed_solution
        return row

    def convert_solution_to_english(row):
        processed_solution = ""
        for char in row["bengali_solution"]:
            if char in "০১২৩৪৫৬৭৮৯":
                mapping = {
                    "০": "0",
                    "১": "1",
                    "২": "2",
                    "৩": "3",
                    "৪": "4",
                    "৫": "5",
                    "৬": "6",
                    "৭": "7",
                    "৮": "8",
                    "৯": "9",
                }
                processed_solution += mapping[char]
            elif char in "0123456789":
                processed_solution += char
            elif char in ".":
                processed_solution += "."
            elif char in ",":
                pass
            else:
                print(f"error: {char}")
                raise ValueError(f"error: {char}")
        row["english_solution"] = processed_solution
        return row

    def convert_problem_to_bengali_number(row):
        processed_problem = ""
        for char in row["problem"]:
            if char in "0123456789":
                mapping = {
                    "0": "০",
                    "1": "১",
                    "2": "২",
                    "3": "৩",
                    "4": "৪",
                    "5": "৫",
                    "6": "৬",
                    "7": "৭",
                    "8": "৮",
                    "9": "৯",
                }
                processed_problem += mapping[char]
            else:
                processed_problem += char
        row["problem"] = processed_problem
        return row

    row = convert_solution_to_bengali(row)
    row = convert_solution_to_english(row)
    row = convert_problem_to_bengali_number(row)
    del row["solution"]
    return row


def filter_dataset(ds):
    # ds = filter_and_print_stat(ds, is_source_includes)
    ds = filter_and_print_stat(ds, is_valid_row)
    ds = filter_and_print_stat(ds, is_solution_verifiable)
    ds = ds.map(process_solution)
    ds = filter_and_print_stat(ds, is_bengali_in_question)
    ds = filter_and_print_stat(ds, is_multiple_choice)
    return ds


if __name__ == "__main__":
    ds = load_dataset("GanitLLM/GANIT-v2")
    ds["train"] = ds["train"].add_column("id", range(len(ds["train"])))
    print(f"Before filtering: {len(ds['train'])}")
    ds["train"] = filter_dataset(ds["train"])
    print(f"After filtering: {len(ds['train'])}")

    print(ds)
    ds.push_to_hub("GanitLLM/GANIT_filtered_v2", private=True)
