from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer, pipeline

import time

start_time_script = time.time()

import sys
import os
import argparse
import random
import json
import re

from openai import OpenAI
import openai
import torch

from tqdm import tqdm

# metrices
import evaluate

parser = argparse.ArgumentParser(description="Benchmarking of Readers")

# Add command line arguments
parser.add_argument("--language", type=str, help="Specify the language")
parser.add_argument("--indexName", type=str, help="Specify the index name")
parser.add_argument("--reader", type=str, help="Specify the reader")
parser.add_argument("--dataset", type=str, help="Specify the dataset to use")
parser.add_argument("--contextLength", type=int, help="Specify the context length")
parser.add_argument("--randomShuffle", type=str, help="Specify if the data should be shuffled randomly")
parser.add_argument("--batchSize", type=int, help="Specify the batch size")
parser.add_argument("--splitIndex", type=int, help="Specify the split index")
parser.add_argument("--split", type=int, help="Split length or None")

# Parse the command line arguments
args = parser.parse_args()

# Access the values of the arguments
language = args.language
index_name = args.indexName
reader = args.reader
dataset = args.dataset
context_length = args.contextLength
random_shuffle = args.randomShuffle
batch_size = args.batchSize
split_index = args.splitIndex
split = args.split

if random_shuffle == "True":
    random_shuffle = True
elif random_shuffle == "False":
    random_shuffle = False
else:
    print("No valid value provided for --randomShuffle argument.")
    sys.exit(1)

print(f"""Running with arguments: 
      
        language: {language}
        index_name: {index_name}
        reader: {reader}
        dataset: {dataset}
        context_length: {context_length}
        random_shuffle: {random_shuffle}

      """)

if not index_name:
    print("No value provided for --indexName argument.")
    sys.exit(1)

if not language:
    print("No value provided for --language argument.")
    sys.exit(1)

index_file_path = os.path.join(os.environ['HOME'], 'code', 'data_augmentation', 'qa_dataset', index_name + ".json")

class BaseReader():
    def __init__(self):
        self.name = "BaseReader"

    def generate_answer(self, question, context):
        '''Generate answer for a question given some context.'''
        return None
    
    def get_predictions(self, input):
        '''Get predictions from a model.'''
        return None

class OpenAIGPT(BaseReader):
    def __init__(self):
        super().__init__()
        self.name = "OpenAI-GPT35"
        self.openai_client = OpenAI(
            api_key="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"
        )

    
    def generate_answer(self, question, context):
        '''Generate answer for a question given some context.'''

        self.system_prompt = "You are an assistant who generates an answer given multiple contexts, which not necessarily contain the answer. Please generate an answer or state, 'I don't know' if you can not generate an answer. If you need to ask clarification questions, feel free to do so."

        answers = []

        # Create a prompt for OpenAI GPT-3.5
        for i in range(len(context)):
            prompt = f"Please generate an answer for the following question: {question[i]} \nUse the following contexts: {self.list_of_strings_to_string(context[i])}\n"
            answers.append(self.get_predictions(prompt))

        return answers

    def get_predictions(self, input_text):
        '''Get predictions from a model.'''
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": input_text},
                ]
            )
            # Extract the similarity score from the model's response using regex
            return response.choices[0].message.content

        except openai.RateLimitError as e:
            # Handle rate limit error gracefully
            print(f"Rate limit exceeded. Waiting for 20 seconds before retrying.")
            time.sleep(20)  # You may adjust the sleep duration
            return self.get_predictions(input_text)

        except Exception as e:
            # Handle other exceptions if needed
            print(f"An error occurred with OpenAI API: {e}")
            return None  # or handle in a way that suits your application
    
    def list_of_strings_to_string(self, list_of_strings):
        listed = ""
        for string in list_of_strings:
            listed += string + "\n\n"
        return listed
    
class Llama2(BaseReader):
    def __init__(self, generate_gold_answers=False, batch_size=1):
        super().__init__()
        self.name = "Llama-2-7B"

        self.generate_gold_answers = generate_gold_answers

        print("Setting up Llama-2")

        use_triton = False

        model_file = "TheBloke/Llama-2-7B-Chat-GPTQ"

        tokenizer = AutoTokenizer.from_pretrained(model_file, padding=True)


        print("Loaded Tokenizer")

        model = AutoGPTQForCausalLM.from_quantized(model_file,
            use_safetensors=True,
            trust_remote_code=True,
            device_map="auto",
            use_triton=use_triton,
            # stop=["[INST]", "None", "User:"]
        )

        # For batch processing
        # tokenizer.pad_token = tokenizer.bos_token
        model.config.pad_token_id = model.config.eos_token_id
        # model.config.pad_token_id = model.config.bos_token_id

        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"

        print("Initialized model")

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=20,
            device_map="auto",
            repetition_penalty=1.1,
            max_new_tokens=546,
            batch_size=batch_size,
        )

        print("Initialized pipeline")

    def benchmarking(self, questions, contexts):
        # Create a dataloader of prompts
        def data():
            if self.generate_gold_answers:
                prompt = f'''
[INST] <<SYS>>
You are an assistant who generates gold answers for a given question and context.
'''
            else:
                prompt = f'''
[INST] <<SYS>>
You are an assistant who generates an answer given multiple contexts, which not necessarily contain the answer. You only respond to the "Assistant". Please generate an answer or state, "I can't provide an answer given the context." if you can not provide an answer based on the provided context.
'''
            for i in range(len(questions)):
                yield prompt + f"\n<</SYS>>Please generate an answer for the following question: {questions[i]} \nUse the following contexts: {self.list_of_strings_to_string(contexts[i])}[/INST]\nAssistant:"


        # Process data through the pipeline
        total_contexts = len(questions)
        
        generated_answers = []

        start_time = time.time()

        # Process data through the pipeline
        generated_characters = 0

        i = 0         
        with tqdm(total=total_contexts, file=sys.stdout) as pbar:
            for generated_text in self.pipe(data(), num_return_sequences=1):
                generated_characters += len(generated_text[0]["generated_text"])
                generated_answers.append(generated_text[0]["generated_text"].split("Assistant:")[-1].strip())

                # Set the progress value for tqdm
                pbar.update(1)
                i += 1

        total_time = time.time() - start_time
        print(f"Total Time: {total_time:.2f} seconds")

        # Return question, context, answer triplets
        # triplets = []
        # for i in range(len(questions)):
        #     triplets.append((questions[i], contexts[i], generated_answers[i]))
        return generated_answers
    
    def list_of_strings_to_string(self, list_of_strings):
        listed = ""
        for string in list_of_strings:
            listed += string + "\n\n"
        return listed

class LeoLama2(BaseReader):
    def __init__(self, generate_gold_answers=False):
        super().__init__()
        self.name = "LeoLama-2-7B"
        self.generate_gold_answers = generate_gold_answers

        print(f"Setting up {self.name}")

        use_triton = False
        model_file = "TheBloke/leo-hessianai-7B-chat-GPTQ"

        tokenizer = AutoTokenizer.from_pretrained(model_file, use_fast=True)

        print("Loaded Tokenizer")

        if self.generate_gold_answers:
            self.system_prompt = """Du bist ein Assistent, der eine Referenzantwort für eine gegebene Frage und Kontext generiert."""
        else:
            self.system_prompt = """Du bist ein Assistent, der aus mehreren Kontexten, die nicht unbedingt die Antwort enthalten, eine Antwort generiert. Bitte beantworte die Frage oder sage "Ich kann keine Antwort geben, da der Kontext fehlt.", wenn du keine Antwort geben kannst basierend auf dem Kontext."""

        model = AutoGPTQForCausalLM.from_quantized(model_file,
            use_safetensors=True,
            trust_remote_code=False,
            #device = 0,
            inject_fused_attention=False,
            device_map="auto",
            use_triton=use_triton
        )

        print("Initialized model")

        self.pipe = pipeline("text-generation",model=model,tokenizer=tokenizer, torch_dtype=torch.float16, trust_remote_code=False) # True for flash-attn2 else False

        print("Initialized pipeline")

    def benchmarking(self, questions, contexts):

        def data():
            prompt_format = "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            for i in range(len(questions)):
                prompt = f"Bitte generiere eine Antwort für die folgende Frage: {questions[i]} \nVerwende die folgenden Kontexte: {self.list_of_strings_to_string(contexts[i])}\n"
                yield prompt_format.format(prompt=prompt, system_prompt=self.system_prompt)
        
        # Process data through the pipeline
        total_contexts = len(questions)
        
        generated_answers = []

        start_time = time.time()

        # Process data through the pipeline
        generated_characters = 0
        
        i = 0
        for generated_text in self.pipe(data(), num_return_sequences=1, do_sample=True, top_p=0.95, max_length=819):
            generated_characters += len(generated_text[0]["generated_text"])
            generated_answers.append(generated_text[0]["generated_text"].split("assistant")[-1].strip())

            elapsed_time = time.time() - start_time

            # Calculate progress percentage (multiply by 2 to reach 100%)
            progress = (i + 1) / (total_contexts) * 100

            # Print progress and elapsed time
            print(f"Progress: {progress:.2f}%, Elapsed Time: {elapsed_time:.2f} seconds, Generated Characters: {generated_characters}")
            
            i += 1
            
        total_time = time.time() - start_time
        print(f"Total Time: {total_time:.2f} seconds")

        # Return question, context, answer triplets
        # triplets = []
        # for i in range(len(questions)):
        #     triplets.append((questions[i], contexts[i], generated_answers[i]))
        # return triplets
        return generated_answers
    
    def list_of_strings_to_string(self, list_of_strings):
        listed = ""
        for string in list_of_strings:
            listed += string + "\n\n"
        return listed

# Function to generate tuples of questions and contexts
def generate_question_context_lists(text_corpus, questions):
    generated_questions = []
    generated_contexts = []

    for i in range(len(questions)):
        for question in questions[i]:
            correct_context = text_corpus[i]

            # Make a list of all available contexts excluding the correct one
            all_contexts = [context for j, context in enumerate(text_corpus) if j != i]

            # Select 4 random contexts from the list
            random_contexts = random.sample(all_contexts, 4)

            # Combine correct context and random contexts
            all_contexts_combined = [correct_context] + random_contexts

            # Shuffle the combined list to randomize the order
            random.shuffle(all_contexts_combined)

            # Append the question and corresponding contexts to the result lists
            generated_questions.append(question)
            generated_contexts.append(all_contexts_combined)

    return generated_questions, generated_contexts

### Benchmarking metrics

# Blue-1

blue = evaluate.load('bleu')
predictions = ["hello there general kenobi", "foo bar foobar"]
references = [["hello there general kenobi", "hello there !"],["foo bar foobar"]]

results = blue.compute(predictions=predictions, references=references)
print(f"Blue-1: {results}")

# Rouge-L

rouge = evaluate.load('rouge')
results = rouge.compute(predictions=predictions, references=references, rouge_types=["rougeL"])
print(f"Rouge-L: {results}")

# F1-BERTScore

bertscore = evaluate.load("bertscore")
results = bertscore.compute(predictions=predictions, references=references, lang=language)
print(f"F1-BERTScore: {results}")

# Accuracy using LLM

def llm_accuracy(questions, predictions, references):
    client = OpenAI(
        api_key="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"
    )

    if len(questions) >= 1000:
        # Get the length of the lists
        list_length = len(questions)

        # Sample 1000 random indices
        if split != 1:
            random_indices = random.sample(range(list_length), 1000//split)
        else:
            random_indices = random.sample(range(list_length), 1000)

        # Extract sampled elements from the lists
        questions = [questions[i] for i in random_indices]
        predictions = [predictions[i] for i in random_indices]
        references = [references[i] for i in random_indices]

    # Iterate through each pair of prediction list and reference
    similarity_scores = []

    for question, prediction, reference in zip(questions, predictions, references):
        # Calculate the similarity score for each prediction in the list
        # Create a prompt for OpenAI GPT-3.5 to compare the prediction and reference
        prompt = f"Question: {question}\nGold Answer: {reference}\nGenerated Answer: {prediction}\nIs the generated answer correct? Give a binary answer Yes (1) or No (0)\n"

        def get_predictions(input_text):
            '''Get predictions from a model.'''
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant who evaluates, if a generated answer is correct in relation to a gold answer."},
                    {"role": "user", "content": input_text},
                ]
            )
            # Extract the similarity score from the model's response using regex
            response_content = response.choices[0].message.content
            return response_content

        try:
            response_content = get_predictions(prompt)

        except openai.RateLimitError as e:
            # Handle rate limit error gracefully
            print(f"Rate limit exceeded. Waiting for 20 seconds before retrying.")
            time.sleep(20)  # You may adjust the sleep duration
            response_content = get_predictions(prompt)

        except Exception as e:
            # Handle other exceptions if needed
            print(f"An error occurred with OpenAI API: {e}")
            return None

        match = re.search(r'1|0', response_content)

        if match:
            similarity_score = int(match.group())
            similarity_scores.append(similarity_score)


    # Calculate the average similarity score for the predictions
    accuracy_scores = sum(similarity_scores) / len(similarity_scores)

    return {"accuracy":accuracy_scores, "similarity_scores":similarity_scores}




### Load Data
if dataset == "qac":
    gold_answers_file_path = os.path.join(os.environ['HOME'], 'code', 'data_augmentation', 'qac_extended_dataset', index_name + ".json")
    with open(gold_answers_file_path, 'r') as file:
        data = json.load(file)
    
    questions = data["questions"]
    text_corpus = [[i[0]] for i in data["contexts"]]
    gold_answers = data["answers"]

elif dataset == "qac_extended":
    gold_answers_file_path = os.path.join(os.environ['HOME'], 'code', 'data_augmentation', 'qac_extended_dataset', index_name + ".json")
    with open(gold_answers_file_path, 'r') as file:
        data = json.load(file)
    
    questions = data["questions"]
    text_corpus = [i for i in data["contexts"]]
    gold_answers = data["answers"]

elif dataset == "generate_qac":
    file_path = os.path.join(os.environ['HOME'], 'code', 'data_augmentation', 'qa_dataset', index_name + ".json")
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    text_corpus = data["context"]
    questions = data["questions"]

    # Flatten the lists and attach corresponding text_corpus elements
    text_corpus = [[text] for text, ques_list in zip(text_corpus, questions) for _ in ques_list]
    questions = [ques for ques_list in questions for ques in ques_list]
    gold_answers = [None] * len(questions)

else:
    print("No valid dataset specified.")
    sys.exit(1)

### TODO: For Debugging only

# # # Comment this lines till total questions out for full benchmark run
# crop = 10

# questions = questions[:crop]
# text_corpus = text_corpus[:crop]
# gold_answers = gold_answers[:crop]

print("Total questions: ", len(questions))

def split_data(data, split, split_index):
    total_length = len(data)
    chunk_size = total_length // split
    start = split_index * chunk_size
    end = (split_index + 1) * chunk_size if split_index < split - 1 else total_length
    return data[start:end]

# Split the data based on the split index
if split != 1:
    # Split questions
    questions = split_data(questions, split, split_index)

    # Split text_corpus
    text_corpus = split_data(text_corpus, split, split_index)

    # Split gold_answers
    gold_answers = split_data(gold_answers, split, split_index)

    print("Total questions after split: ", len(questions))

### Prepare Data
if dataset != "generate_qac":
    print("Preparing data...")
    start_time = time.time()  # Record start time

    # Remove in 20% of cases the correct context and change the gold answer to "I can't provide an answer given the context."
    if dataset != "qac":
        print("Dropping context...")
        dropped_contexts = []
        for i in range(len(text_corpus)):
            # Check if we should remove the correct context
            if random.random() < 0.2:
                # Remove the correct context
                text_corpus[i] = text_corpus[i][1:]

                # Update the gold answer
                if language == "de":
                    gold_answers[i] = "Ich kann keine Antwort geben, da der Kontext fehlt."
                elif language == "en":
                    gold_answers[i] = "I can't provide an answer given the context."

                # Add the dropped context index to the list
                dropped_contexts.append(i)

    # Crop to context length
    text_corpus = [i[:context_length] for i in text_corpus]

    # Shuffle the data
    if random_shuffle:
        print("Shuffling data...")
        for i in range(len(text_corpus)):
            random.shuffle(text_corpus[i])

    print(f"Data preparation completed in {time.time() - start_time} seconds.\n")

### Readers
print(f"Using {reader} reader...")
start_time = time.time()  # Record start time

if dataset == "generate_qac" or dataset == "qac":
    generate_gold_answers = True
else:
    generate_gold_answers = False

if reader == "Llama2":
    reader = Llama2(generate_gold_answers=generate_gold_answers, batch_size=batch_size)

    # questions, text_corpus = generate_question_context_lists(text_corpus, questions)

    answers = reader.benchmarking(questions, text_corpus)

elif reader == "LeoLama2":
    reader = LeoLama2(generate_gold_answers=generate_gold_answers)

    # questions, text_corpus = generate_question_context_lists(text_corpus, questions)

    answers = reader.benchmarking(questions, text_corpus)

elif reader == "GPT3":
    reader = OpenAIGPT()

    # questions, text_corpus = generate_question_context_lists(text_corpus, questions)

    # sample only 1000 

    # Get the length of the lists
    list_length = len(questions)

    # Sample 500 random indices
    random_indices = random.sample(range(list_length), 1000)

    # Update Dropped Contexts
    dropped_contexts = [random_indices.index(i) for i in dropped_contexts if i in random_indices]

    # Extract sampled elements from the lists
    questions = [questions[i] for i in random_indices]
    text_corpus = [text_corpus[i] for i in random_indices]
    gold_answers = [gold_answers[i] for i in random_indices]

    answers = reader.generate_answer(questions, text_corpus)

else:
    print("No valid reader specified.")
    sys.exit(1)

print(f"Reader prediction completed in {time.time() - start_time} seconds.\n")

if dataset == "generate_qac":
    if split != 1:
        # save_path = os.path.join(os.environ['HOME'], 'code', 'data_augmentation', 'qac_dataset', index_name + f"_{batch_size}" + "_long.json")
        save_path = os.path.join(os.environ['HOME'], 'code', 'data_augmentation', 'qac_dataset', index_name + "_long" + f"_{split_index}" + ".json")

    else:
        save_path = os.path.join(os.environ['HOME'], 'code', 'data_augmentation', 'qac_dataset', index_name + "_long.json")

    content = {
        "questions": questions,
        "contexts": text_corpus,
        "answers": answers
    }

    with open(save_path, 'w') as file:
        json.dump(content, file, indent=4)

    print("Saved generated data to file: ", save_path)
    print("Exiting...")
    exit()

if split != 1:
    predictions_file_path = os.path.join(os.environ['HOME'], 'code', 'reader', 'data', language, index_name, reader.name, dataset, str(context_length), str(random_shuffle), f"{reader.name}_{index_name}_{dataset}_{split_index}.json")
else:
    # Save predictions to file
    predictions_file_path = os.path.join(os.environ['HOME'], 'code', 'reader', 'data', language, index_name, reader.name, dataset, str(context_length), str(random_shuffle), f"{reader.name}_{index_name}_{dataset}.json") 

# Check if the directory exists if not create it
print("Saving predictions to file...")
os.makedirs(os.path.dirname(predictions_file_path), exist_ok=True)

# Store to json the four lists: questions, contexts, answers, gold_answers

content = {
    "questions": questions,
    "contexts": text_corpus,
    "answers": gold_answers,
    "generated_answers": answers 

}

with open(predictions_file_path, 'w') as file:
    json.dump(content, file, indent=4)

# Calculate metrics
blue = evaluate.load('bleu')
rouge = evaluate.load('rouge')
bertscore = evaluate.load("bertscore")

print("Calculating metrics...")
start_time = time.time()  # Record start time

# split answers, questions and gold_answers in two lists based on the dropped_contexts list

# store the dropped_contexts index answers, questions and gold_answers in a new list
answers_dropped_contexts = [answers[i] for i in dropped_contexts]
questions_dropped_contexts = [questions[i] for i in dropped_contexts]
gold_answers_dropped_contexts = [gold_answers[i] for i in dropped_contexts]

# store non dropped_contexts index answers, questions and gold_answers in a new list
answers = [answers[i] for i in range(len(answers)) if i not in dropped_contexts]
questions = [questions[i] for i in range(len(questions)) if i not in dropped_contexts]
gold_answers = [gold_answers[i] for i in range(len(gold_answers)) if i not in dropped_contexts]

# Calculate Blue-1
blue_results = blue.compute(predictions=answers, references=gold_answers)

# Calculate Rouge-L
rouge_results = rouge.compute(predictions=answers, references=gold_answers, rouge_types=["rougeL"])

# Calculate F1-BERTScore
bertscore_results = bertscore.compute(predictions=answers, references=gold_answers, lang=language)


# TODO: For now I leave this one out because it costs money
# Calculate LLM Accuracy
try:
    llm_accuracy_results = llm_accuracy(questions=questions, predictions=answers, references=gold_answers)
except Exception as e:
    print("Exception occured while calculating LLM Accuracy:", e)
    llm_accuracy_results = "None"



# Calculate all metrics for dropped contexts

# Calculate Blue-1
blue_results_dropped_contexts = blue.compute(predictions=answers_dropped_contexts, references=gold_answers_dropped_contexts)

# Calculate Rouge-L
rouge_results_dropped_contexts = rouge.compute(predictions=answers_dropped_contexts, references=gold_answers_dropped_contexts, rouge_types=["rougeL"])

# Calculate F1-BERTScore
bertscore_results_dropped_contexts = bertscore.compute(predictions=answers_dropped_contexts, references=gold_answers_dropped_contexts, lang=language)


try:
    llm_accuracy_results_dropped_contexts = llm_accuracy(questions=questions_dropped_contexts, predictions=answers_dropped_contexts, references=gold_answers_dropped_contexts)
except Exception as e:
    print("Exception occured while calculating LLM Accuracy:", e)
    llm_accuracy_results_dropped_contexts = "None"


print(f"Metrics calculated based on {len(questions)} questions and {len(dropped_contexts)} dropped contexts in {time.time() - start_time} seconds.\n")

if split != 1:
    metrics_file_path = os.path.join(os.environ['HOME'], 'code', 'reader', 'benchmarks', language, index_name, reader.name, dataset, str(context_length), str(random_shuffle), f"{reader.name}_{index_name}_{dataset}_{split_index}_metrics.json")
else:
    # Save metrics to file
    metrics_file_path = os.path.join(os.environ['HOME'], 'code', 'reader', 'benchmarks', language, index_name, reader.name, dataset, str(context_length), str(random_shuffle), f"{reader.name}_{index_name}_{dataset}_metrics.json")

# Check if the directory exists if not create it
os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)

content = {
    "all": {
        "amount" : len(questions),
        "blue": blue_results,
        "rouge": rouge_results,
        "bertscore": bertscore_results,
        "llm_accuracy": llm_accuracy_results
    }
}

if dataset != "qac":
    content["dropped_contexts"] = {
        "amount": len(dropped_contexts),
        "blue": blue_results_dropped_contexts,
        "rouge": rouge_results_dropped_contexts,
        "bertscore": bertscore_results_dropped_contexts,
        "llm_accuracy": llm_accuracy_results_dropped_contexts
    }

print("Saving metrics to file...")

with open(metrics_file_path, 'w') as file:
    json.dump(content, file, indent=4)

print(f"Script took {time.time() - start_time_script} seconds")