import json
import os
import argparse

def get_order(filename):
    # Extrahieren der letzten Ganzzahl aus dem Dateinamen
    return int(filename.split('_')[-1].split('.')[0])


def accumulate_jsons(directory, index_name):
    accumulated_data = {}
    questions = []
    answers = []
    contexts = []

    question_count = 0
    answer_count = 0
    context_count = 0

    # All relevant files
    relevant_files = [filename for filename in os.listdir(directory) if filename.endswith(".json") and filename.startswith(index_name)]

    for filename in sorted(relevant_files, key=get_order):
        print(f"Loading file: {filename}")
        with open(os.path.join(directory, filename)) as file:
            data = json.load(file)
            question = data["questions"]
            answer = data["answers"]
            context = data["contexts"]

            question_count += len(question)
            answer_count += len(answer)
            context_count += len(context)

            questions.extend(question)
            answers.extend(answer)
            contexts.extend(context)

    accumulated_data["questions"] = questions
    accumulated_data["answers"] = answers
    accumulated_data["contexts"] = contexts

    # Speichern der akkumulierten Daten in einer neuen JSON-Datei
    with open(os.path.join(directory, index_name + ".json"), 'w') as outfile:
        print(f"Saving accumulated data to: {index_name}.json")
        json.dump(accumulated_data, outfile, indent=4)
    
    return question_count, answer_count, context_count

# ArgumentParser-Objekt erstellen
parser = argparse.ArgumentParser(description='Accumulate JSON files based on index name.')
# Argument hinzufügen
parser.add_argument('--indexName', type=str, help='The index name to be included in the filename.')
args = parser.parse_args()

print(f"Index name: {args.indexName}")

directory = "/home/hd/hd_hd/hd_ff305/code/data_augmentation/qac_dataset"
index_name = args.indexName
question_count, answer_count, context_count = accumulate_jsons(directory, index_name)

# Please load the created accumulated_data file and compare the number of questions, answers and contexts with the original data.
# Laden der erstellten akkumulierten Daten
with open(os.path.join(directory, index_name + ".json"), 'r') as infile:
    loaded_data = json.load(infile)

# Vergleichen der Anzahl von Fragen, Antworten und Kontexten
print(f"Original questions: {question_count}, Loaded questions: {len(loaded_data['questions'])}")
print(f"Original answers: {answer_count}, Loaded answers: {len(loaded_data['answers'])}")
print(f"Original contexts: {context_count}, Loaded contexts: {len(loaded_data['contexts'])}")
