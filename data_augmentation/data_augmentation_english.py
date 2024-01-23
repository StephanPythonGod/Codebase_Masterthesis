import pymongo
import platform

print("Starting Python Script ...")
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

index_name = "IndexEnglish"
index_english = download_index(index_name)
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

def generate_questions_for_index(index, examples, pipeline, save_path, num_questions):
    # Extract the passages from the index
    documents = index.get("documents", [])
    contexts = []
    for document in documents:
        contexts += document.get("passages", [])

    generated_question_tuples = []

    # Create a list of prompts
    for i, (example_context, example_question, example_answer) in enumerate(examples):
        prompt = f'''
[INST] <<SYS>>
You are an assistant who generates one question given a context. Please start your generated question with the indicator 'Generated Question:' Here are some examples:

Example {i + 1}:
Context: {example_context}
Generated Question: {example_question}

'''
    def data():
        for context in contexts:
            yield prompt + f"<<SYS>>\nGenerate the question based on the following context:\n\nContext: {context}[/INST]"
    
    total_contexts = len(contexts)

    start_time = time.time()

    # Process data through the pipeline
    generated_characters = 0
    
    i = 0
    for generated_text in pipeline(data(), num_return_sequences=1):
        question = []
        generated_characters += len(generated_text[0]["generated_text"])
        question.append(generated_text[0]["generated_text"].split("Generated Question:")[-1].strip())
            
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
model_file = "TheBloke/Llama-2-7B-Chat-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(model_file, use_fast=True)
print("Loaded Tokenizer")

model = AutoGPTQForCausalLM.from_quantized(model_file,
    use_safetensors=True,
    trust_remote_code=True,
    device_map="auto",
    use_triton=use_triton
)

print("Initialized model")

pipe = pipeline(
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
)

example1 = ("The master's program in Data and Computer Science includes an application area. Annex 3 lists the possible application areas. Upon request, the examination board can also approve a different application area. - Master Data and Computer Science",
            "I study Data and Computerscience. Can I choose application areas that are not listed?",
            "Yes, upon request, the examination board can also approve a different application area.")

example2 = ("(5) The final failure in a mandatory module leads to the loss of the examination claim. In elective mandatory modules, if provided for in the module handbook, the failure can be compensated for by the successful completion of another elective mandatory module or another performance within the respective module. § 4 Paragraph 2 remains unaffected. - Bachelor Physics",
            "When does the failure of an exam lead to the termination of the study?",
            "For Bachelor Students of Phisics, the final failure in a mandatory module leads to the loss of the examination claim. In elective mandatory modules, this can be compensated for.")

example3 = ("(3) The application must be made in writing to the examination board. It is the responsibility of the applicant to provide the necessary information about the performance to be recognized. The burden of proof for the existence of a significant difference in academic achievements lies with Heidelberg University; the obligations of the applicant, especially according to Sentence 1 and Sentence 2, remain unaffected. The burden of proof for the existence of equivalence in non-academic achievements lies with the applicant. - Bachelor Philosophy",
            "How can I obtain recognition of study achievements in Heidelberg?",
            "Recognition of study achievements requires a written application to the examination board. The applicant is responsible for providing all necessary information regarding the performance to be recognized.")

example4 = ("The intermediate examination consists of successful participation in the exercises for beginners in the subjects Civil Law, Public Law, and Criminal Law. The partial performances of the exercise (homework and supervisory work under examination conditions) must in principle be performed in the exercise of a semester; § 4 paragraph 5 remains unaffected.",
            "What are the components of the intermediate examination in Civil Law, Public Law, and Criminal Law?",
            "The intermediate examination in Civil Law, Public Law, and Criminal Law consists of successful participation in exercises for beginners, which includes completing homework and supervisory work under examination conditions during the semester.")


examples = [example1, example2, example3, example4]

import os

base_destination_path = os.path.join(os.environ['HOME'], 'code', 'data_augmentation', 'qa_dataset')

# In your Python script, access the array index passed as a command-line argument
import sys

# Get the array index from the command-line arguments
ARRAY_INDEX = int(sys.argv[1])

for i in range(1, 4):  # Run the loop 3 times
    print(f"Starting the {i} round for index: {index_name}")
    
    # Update the "database" field in your index dictionary
    index = index_english  # or index_german, depending on your use case
    index["database"] = f"{index_name}{ARRAY_INDEX}"  # Use ARRAY_INDEX
    
    qa_dataset_path = generate_questions_for_index(index, examples, pipe, base_destination_path, i)
