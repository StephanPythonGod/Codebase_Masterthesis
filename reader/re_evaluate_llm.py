from openai import OpenAI
import openai
import sys
import os
import argparse
import random
import json
import re
import time


def llm_accuracy(questions, predictions, references, split):
    client = OpenAI(
        api_key="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"
    )

    if len(questions) >= 200:
        # Get the length of the lists
        list_length = len(questions)

        # Sample 1000 random indices
        if split != 1:
            random_indices = random.sample(range(list_length), 200//split)
        else:
            random_indices = random.sample(range(list_length), 200)

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
            return "An Error occurred"

        match = re.search(r'1|0', response_content)

        if match:
            similarity_score = int(match.group())
            similarity_scores.append(similarity_score)


    # Calculate the average similarity score for the predictions
    accuracy_scores = sum(similarity_scores) / len(similarity_scores)

    return {"accuracy":accuracy_scores, "similarity_scores":similarity_scores}

important_indices = ["translated_IndexEnglishAllFiltered_long", "IndexEnglishAllFiltered_long"]
filtered_paths = []

for root, dirs, files in os.walk('./benchmarks'):
    if not dirs:  # This means `root` is a leaf directory
        if any(index in root for index in important_indices):
            for file in files:
                full_path = os.path.join(root, file)
                with open(full_path, 'r') as f:
                    data = json.load(f)
                    values = data.values()
                    for value in values:
                        if isinstance(value['llm_accuracy'], str):
                            filtered_paths.append(root)

filtered_paths = set(filtered_paths)

print(filtered_paths)

for path in filtered_paths:
    # check for data files in the same path, except that the root folder is now ./data
    path = path.replace('./benchmarks', './data')
    # get all files in path
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith('.json'):
                print(f"Split: {file.split('_')[-1][0]}")
                try:
                    if file.split("_")[-1][0] in ["0", "1", "2", "3", "4", "5"]:
                        split = 5
                    else:
                        split = 1
                except:
                    split = 1
                full_path = os.path.join(path, file)
                with open(full_path, 'r') as f:
                    data = json.load(f)
                    # Now `data` contains the data from the JSON file
                    questions = data['questions']
                    answers = data['answers']
                    contexts = data['contexts']
                    predictions = data['generated_answers']

                    # Filter dropped_context and not
                    if path.split("/")[2] == "en":
                        dropped_answer = "I can't provide an answer given the context."
                    else:
                        dropped_answer = "Ich kann keine Antwort geben, da der Kontext fehlt."

                    # Get all indices of dropped_context based on answers
                    dropped_indices = [i for i, x in enumerate(answers) if x == dropped_answer]

                    # Create new lists without dropped_context
                    filtered_questions = [i for j, i in enumerate(questions) if j not in dropped_indices]
                    filtered_predictions = [i for j, i in enumerate(predictions) if j not in dropped_indices]
                    filtered_contexts = [i for j, i in enumerate(contexts) if j not in dropped_indices]

                    # Create new lists with only dropped_context
                    dropped_questions = [i for j, i in enumerate(questions) if j in dropped_indices]
                    dropped_predictions = [i for j, i in enumerate(predictions) if j in dropped_indices]
                    dropped_contexts = [i for j, i in enumerate(contexts) if j in dropped_indices]

                    # Get the predictions from the model
                    accuracy = llm_accuracy(filtered_questions, filtered_predictions, filtered_contexts, split)
                    accuracy_dropped = llm_accuracy(dropped_questions, dropped_predictions, dropped_contexts, split)

                    # Add the accuracy values to the corresponding metrics files
                    metric_path = full_path.replace("data", "benchmarks")
                    metric_path = metric_path.replace(".json", "_metrics.json")
                    print(metric_path)
                    print(f"Questions: {len(filtered_questions)}")
                    print(f"Dropped: {len(dropped_questions)}")
                    print(f"Split: {split}")
                    print()

                    # accuracy = "TestAccuracy"

                    # accuracy_dropped = "TestAccuracyDropped"

                    with open(metric_path, 'r') as f:
                        metrics = json.load(f)

                        for key, value in metrics.items():
                            if key == "all":
                                metrics[key]["llm_accuracy"] = accuracy
                            else:
                                metrics[key]["llm_accuracy"] = accuracy_dropped

                        # Close the file
                        f.close()
                    
                    with open(metric_path, 'w') as f:
                        json.dump(metrics, f, indent=4)
