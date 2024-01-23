import sys
import os
import json

from tqdm import tqdm
from openai import OpenAI
import time
import random

import concurrent.futures

import argparse

parser = argparse.ArgumentParser(description="Generate gold answers for the given questions and contexts")

parser.add_argument("--indexName", type=str, help="Name of the index to use")
parser.add_argument("--language", type=str, help="Language")

args = parser.parse_args()

index_name = args.indexName
language = args.language

if language not in ["en", "de"]:
    print("Language must be either 'en' or 'de'.")
    sys.exit(1)

index_file_path = os.path.join(os.environ['HOME'], 'code', 'data_augmentation', 'qa_dataset', index_name + ".json")

print("Start Loading Data ...")

with open(index_file_path, 'r') as file:
    data = json.load(file)

text_corpus = [item["context"] for item in data]
questions = [item["questions"] for item in data]

# Flatten the lists and attach corresponding text_corpus elements
flattened_text_corpus = [text for text, ques_list in zip(text_corpus, questions) for _ in ques_list]
flattened_questions = [ques for ques_list in questions for ques in ques_list]


# Random sample 2000
random.seed(42)
random_indices = random.sample(range(len(flattened_text_corpus)), 2000)
text_corpus = [flattened_text_corpus[i] for i in random_indices]
questions = [flattened_questions[i] for i in random_indices]

# # Crop the data to 10 entries for testing purposes
# text_corpus = text_corpus[:10]
# questions = questions[:10]

total_questions = sum(len(question_set) for question_set in questions)

print(f"Number Text Corpus Entries: {len(text_corpus)} | Number Questions: {total_questions}")

print(f"Loaded Data.") #takes 7.590401887893677 min

print("Initialize OpenAI API ...")

openai_api_key = "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"

openai_client = OpenAI(
    api_key=openai_api_key,
)

if language == "en":
    def generate_gold_answer(question, context):
        system_prompt = "You are an assistant who generates gold answers for a given question and context.\n"
        prompt = f"Question: {question}\nContext: {context}\n"
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content
elif language == "de":
    def generate_gold_answer(question, context):
        system_prompt = "Du bist ein Assistent der Referenzantwort für eine gegebene Frage und Kontext generiert.\n"
        prompt = f"Frage: {question}\nKontext: {context}\n"
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content

print("Initialized OpenAI API.")

tripplets = []

def process_question(i, questions, text_corpus, tripplets):
    try:
        answer = generate_gold_answer(questions[i], text_corpus[i])
        tripplets.append((questions[i], text_corpus[i], answer))
    except Exception as e:
        print(e)

total_tasks = len(text_corpus)

# tqdm progress bar
with tqdm(total=total_tasks, desc="Processing Questions", file=sys.stdout) as pbar:
    # Process questions sequentially without parallelism
    for i in range(len(text_corpus)):
        # Process each question
        process_question(i, questions, text_corpus, tripplets)
        # Update progress bar
        pbar.update(1)

# Unpack the tripplets list
unpacked_tripplets = list(zip(*tripplets))

# Save the unpacked tripplets to a JSON file
output_data = {"questions": unpacked_tripplets[0], "contexts": unpacked_tripplets[1], "answers": unpacked_tripplets[2]}

tripplets_file_path = os.path.join(os.environ['HOME'], 'code', 'data_augmentation', 'qac_dataset', index_name + "_short.json")

with open(tripplets_file_path, 'w') as file:
    json.dump(output_data, file, indent=4)