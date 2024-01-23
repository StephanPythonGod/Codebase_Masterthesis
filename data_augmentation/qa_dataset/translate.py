import json
from google.cloud import translate_v2 as translate

# Replace 'YOUR_API_KEY' with your Google Translate API key
translate_client = translate.Client()

file_name = "IndexEnglishAll"

def translate_to_german(text):
    translation = translate_client.translate(text, target_language="de")
    return translation["translatedText"]

print("Started Loading Data ...")
# Load the JSON file
with open(f'{file_name}.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

print("Loaded Data")

# Translate the context and questions in each entry
for entry in data:
    entry['context'] = translate_to_german(entry['context'])
    entry['questions'] = [translate_to_german(question) for question in entry['questions']]

# Save the translated data to a new JSON file
with open(f'translated_{file_name}.json', 'w', encoding='utf-8') as output_file:
    json.dump(data, output_file, ensure_ascii=False, indent=4)

print("Translation to German completed. Data saved to 'translated_output.json'.")
