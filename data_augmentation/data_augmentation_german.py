import os
import sys

import pymongo
import platform

print("Starting Python Script")

def download_index(index_name):
    # Define MongoDB connection settings
    username = "google_colab"
    password = "TEBvOpN9x3n5TZhE"
    cluster_url = "masterthesis.2tj9bf6.mongodb.net"

    # Construct the MongoDB connection string
    connection_string = f"mongodb+srv://{username}:{password}@{cluster_url}/test?retryWrites=true&w=majority"

    client = pymongo.MongoClient(connection_string)
    try:
        # Create a MongoDB client

        # Access the specified database
        database_name = index_name  # Assume the database name is the same as the index name
        db = client[database_name]

        # Retrieve all documents from the database
        document_entries = {}

        for document in db["documents"].find({}):
            document_name = document["documentName"]
            document_id = document["documentID"]

            if (document_name, document_id) not in document_entries:
                document_entries[(document_name, document_id)] = {
                    "documentName": document_name,
                    "documentID": document_id,
                    "passages": []
                }

            # Apply the pipeline to the "context" and append to passages
            document_entries[(document_name, document_id)]["passages"].append(document["context"])

        # Convert the dictionary values to a list of document entries
        documents = list(document_entries.values())

        # Construct the index object
        index = {
            "database": database_name,
            "description": db["metadata"].find_one()["description"],
            "documents": documents
        }

        return index

    except pymongo.errors.ConnectionFailure as e:
        print(f"MongoDB Connection Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'client' in locals():
            client.close()

index_name = "IndexGerman"
index_german = download_index(index_name)


print("Received Index from MongoDB")

#Print Python and Cuda version
print("Python Version:", platform.python_version())
try:
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_version = torch.version.cuda
        print("CUDA Available:", cuda_available)
        print("CUDA Version:", cuda_version)
    else:
        print("CUDA Not Available")
except ImportError:
    print("PyTorch not installed, unable to check CUDA version")

import json
import time
import torch

def generate_questions_for_index_german(index, examples, pipeline, save_path, num_questions):
    # Extract the passages from the index
    documents = index.get("documents", [])
    contexts = []
    for document in documents:
        contexts += document.get("passages", [])

    generated_question_tuples = []

    system_prompt = """<|im_start|>system
Du bist ein Assistent, der eine Frage zu einem bestimmten Kontext erstellt. Bitte beginnen deine generierte Frage mit dem Indikator "Generierte Frage:"! Das ist wichtig. Hier sind einige Beispiele:
"""
    for i, (example_context, example_question, example_answer) in enumerate(examples):
        system_prompt += f"Beispiel {i + 1}:\nKontext: {example_context}\nGenerierte Frage: {example_question}\n\n"
    system_prompt += "<|im_end|>"
    
    prompt_format = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    def data():
        for context in contexts:
            prompt = f"Deine Aufgabe ist eine Frage basierend auf folgendem Kontext zu generieren:\n\nKontext: {context}"
            yield prompt_format.format(prompt=prompt)
    
    total_contexts = len(contexts)

    start_time = time.time()

    # Process data through the pipeline
    generated_characters = 0
    
    i = 0
    for generated_text in pipeline(data(), num_return_sequences=1, do_sample=True, top_p=0.95, max_length=8192):
        question = []
        generated_characters += len(generated_text[0]["generated_text"])
        question_tmp = generated_text[0]["generated_text"].split("assistant")[-1].strip().replace("Frage","").strip()
        question_tmp = question_tmp.split("?")[0]
        question_tmp += "?"
        question.append(question_tmp)
            
        generated_question_tuples.append((contexts[i], question))

        elapsed_time = time.time() - start_time

        # Calculate progress percentage (multiply by 2 to reach 100%)
        progress = (i + 1) / (total_contexts) * 100

        # Print progress and elapsed time
        print(f"Progress: {progress:.2f}%, Elapsed Time: {elapsed_time:.2f} seconds, Generated Characters: {generated_characters}")
        
#        torch.cuda.empty_cache()
        i += 1
        
    total_time = time.time() - start_time

    # Organize the data into a list of dictionaries
    dataset = [{"context": passage, "question": question} for (passage, question) in generated_question_tuples]

    # Save the dataset to a JSON file
    file_name = index.get("database", "index") + ".json"
    file_path = f"{save_path}/{file_name}"
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(dataset, json_file, ensure_ascii=False, indent=4)

    print(f"Question-Context dataset saved to: {file_path}")
    print(f"Total Time: {total_time:.2f} seconds")
    return file_path

from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer, pipeline

use_triton = False
model_file = "TheBloke/leo-hessianai-7B-chat-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(model_file, use_fast=True)
print("Loaded Tokenizer")

model = AutoGPTQForCausalLM.from_quantized(model_file,
    use_safetensors=True,
    trust_remote_code=False,
    #device = 0,
    inject_fused_attention=False,
    device_map="auto",
    use_triton=use_triton
)

print("Initialized model")

pipe = pipeline("text-generation",model=model,tokenizer=tokenizer, torch_dtype=torch.float16, trust_remote_code=False) # True for flash-attn2 else False

example1 = ("Das Masterstudium der Data and Computer Science beinhaltet ein Anwendungsgebiet. Anlage 3 listet die möglichen Anwendungsgebiete auf. Auf Antrag kann der Prüfungsausschuss statt diesen auch ein anderes Anwendungsgebiet genehmigen.", "Kann ich Anwendungsgebiete belegen, die nicht aufgelistet sind?", "Ja, auf Antrag kann der Prüfungsausschuss auch ein anderes Anwendungsgebiet genehmigen.")
example2 = ("(5) Das endgültige Nichtbestehen eines Pflichtmoduls führt zum Verlust des Prüfungsan- spruchs. Bei Wahlpflichtmodulen kann, soweit dies im Modulhandbuch vorgesehen ist, das Nichtbestehen durch die erfolgreiche Absolvierung eines anderen Wahlpflichtmoduls oder einer anderen Leistung innerhalb des betreffenden Moduls ausgeglichen werden. § 4 Absatz 2 bleibt unberührt.", "Wann führt eine nicht bestandene Prüfung zum Abbruch des Studiums?", "Das endgültige Nichtbestehen eines Pflichtmoduls führt zum Verlust des Prüfungsanspruchs. Bei Wahlpflichtmodulen kann dies ausgeglichen werden.")
example3 = (
    "(3) Der Antrag ist schriftlich beim Prüfungsausschuss zu stellen. Es obliegt der antragstellenden Person, die erforderlichen Informationen über die anzuerkennende Leistung bereitzustellen. Die Darlegungs- und Beweislast für das Vorliegen eines wesentlichen Unterschieds bei hochschulischen Leistungen liegt bei der Ruprecht-Karls-Universität Heidelberg; Mitwirkungspflichten der antragstellenden Person, insbesondere nach Satz 1 und Satz 2, bleiben hiervon unberührt. Die Darlegungs- und Beweislast für das Vorliegen von Gleichwertigkeit bei außerhochschulischen Leistungen liegen bei der antragstellenden Person.",
    "Wie erhalte ich die Anerkennung von Studienleistungen in Heidelberg?",
    "Die Anerkennung von Studienleistungen erfordert einen schriftlichen Antrag beim Prüfungsausschuss. Die antragstellende Person ist dafür verantwortlich, alle erforderlichen Informationen bezüglich der anzuerkennenden Leistung bereitzustellen."
)
example4 = ("Die Zwischenprüfung besteht aus erfolgreicher Teilnahme an den Übungen für Anfänger in den Fächern Zivilrecht, Öffentliches Recht und Strafrecht. Die Teilleistungen der Übung (Hausaufgaben und betreute Arbeiten unter Prüfungsbedingungen) müssen im Prinzip im Verlauf eines Semesters erbracht werden; § 4 Absatz 5 bleibt unberührt.",
            "Welche Komponenten gehören zur Zwischenprüfung in Zivilrecht, Öffentliches Recht und Strafrecht?",
            "Die Zwischenprüfung in Zivilrecht, Öffentliches Recht und Strafrecht besteht aus erfolgreicher Teilnahme an Übungen für Anfänger, was das Anfertigen von Hausaufgaben und betreuten Arbeiten unter Prüfungsbedingungen während des Semesters einschließt.")


examples = [example1, example2, example3, example4]

base_destination_path = os.path.join(os.environ['HOME'], 'code', 'data_augmentation', 'qa_dataset')

# Get the array index from the command-line arguments
ARRAY_INDEX = int(sys.argv[1])

for i in range(1, 4):  # Run the loop 3 times
    print(f"Starting the {i} round for index: {index_name}")
    
    # Update the "database" field in your index dictionary
    index = index_german  # or index_german, depending on your use case
    index["database"] = f"{index_name}{ARRAY_INDEX}"  # Use ARRAY_INDEX
    
    qa_dataset_path = generate_questions_for_index_german(index, examples, pipe, base_destination_path, 1)