import json
import os

name_of_index = "IndexGerman"
# Directory containing your JSON files
directory = '.'
output_file = f'{name_of_index}All.json'
combined_data = []

questions_by_context = {}

# Iterate through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json") and filename.startswith(name_of_index):
        with open(os.path.join(directory, filename), 'r') as file:
            data = json.load(file)
            for item in data:
                context = item['context']
                question = item['question'][0]  # Assuming there's only one question per item
                if context not in questions_by_context:
                    questions_by_context[context] = []
                questions_by_context[context].append(question)

# Convert the questions_by_context dictionary into the desired format
for context, questions in questions_by_context.items():
    combined_data.append({
        'context': context,
        'questions': questions
    })
# Write the combined data to a single JSON file
with open(output_file, 'w') as output:
    json.dump(combined_data, output, indent=4)

print(f'Combined JSON data written to {output_file}')
