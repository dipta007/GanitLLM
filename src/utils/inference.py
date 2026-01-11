import re
import string
from transformers import AutoTokenizer, AutoModelForCausalLM


def number_of_words(response):
    response = response.strip()
    response = re.sub(r"\s+", " ", response)
    return len(response.split())


def get_percentage_of_bengali(response):
    response = re.sub(r"\s+", "", response)
    bengali_chars = 0
    total_chars = len(response)
    for char in response:
        if (
            char
            in "অআইঈউঊঋঌএঐঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ঺঻়ঽািীুূৃৄেৈোৌ্ৎৗড়ঢ়য়ৠৡৢৣ০১২৩৪৫৬৭৮৯।ওংঃ×÷"
            + string.punctuation
        ):
            bengali_chars += 1
    percentage = bengali_chars / max(total_chars, 1)
    return round(percentage * 100, 2)


messages = [
    {
        "content": 'A conversation takes place between the user and the assistant. The user asks a question, and the assistant solves the problem. Please reason step by step, and put your final answer in the <answer> </answer> tags. Reasoning and answer must be in Bengali.\n\nQuestion: তিন বন্ধু টিভির সামনে বসে আছে।  জানা গেছে, তাদের প্রত্যেকেই সবসময় সঠিক কথা বলে অথবা সবসময় ভুল কথা বলে।  প্রথমজন বললো:  "আমাদের কারোরই এই ছবিটি দেখা হয়নি।"  দ্বিতীয়জন বললো:  "আমি এই ছবিটি দেখেছি, কিন্তু তোমরা দুজনই দেখিনি।"  তৃতীয়জন বললো:  "আমি এই ছবিটি দেখেছি।"  যদি জানা থাকে যে তাদের মধ্যে অন্তত একজন সবসময় সঠিক কথা বলে এবং অন্তত একজন সবসময় ভুল কথা বলে, তাহলে তাদের মধ্যে কতজন সবসময় সঠিক কথা বলে?\n',
        "role": "user",
    },
]

models = [
    "Qwen/Qwen3-4B",
    "dipta007/GanitLLM-4B-SFT-432",
    "dipta007/GanitLLM-4B_CGRPO",
    "dipta007/GanitLLM-4B_SFT_GRPO",
    "dipta007/GanitLLM-4B_SFT_CGRPO",
]

print("Problem:\n", messages[0]["content"])

print("=" * 100)
print("=" * 100)

for model_id in models:
    print(f"Evaluating {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=16384, do_sample=False)

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    # print(response)
    reasoning_content = response.split("<think>")[-1].split("</think>")[0].strip()
    answer = response.split("<answer>")[-1].split("</answer>")[0].strip()

    print("=" * 100)
    print("=" * 100)

    print(f"{model_id} Reasoning content:\n{reasoning_content}")
    print(f"{model_id} Answer:\n{answer}")

    print("=" * 100)
    print("=" * 100)
    print(f"{model_id} Number of words:\n{number_of_words(reasoning_content)}")
    print(
        f"{model_id} Percentage of Bengali:\n{get_percentage_of_bengali(reasoning_content)}"
    )

    print("=" * 100)
    print("=" * 100)
    print("=" * 100)
    print("=" * 100)
