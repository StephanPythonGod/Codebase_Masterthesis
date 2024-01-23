print ("Starting Script")
import sys
import os
import json

from openai import OpenAI
import openai
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


import numpy as np
from tqdm import tqdm

import time

start_time_script = time.time()


### Receive Parameters from bash script
import argparse

parser = argparse.ArgumentParser(description="Retrieve Contexts for the given questions and contexts")

# Add command line arguments
parser.add_argument("--language", type=str, help="Specify the language")
parser.add_argument("--indexName", type=str, help="Specify the index name")

# Parse the command line arguments
args = parser.parse_args()

language = args.language
index_name = args.indexName

print("Language:", language)
print("Index Name:", index_name)

class BaseRetriever:
    def __init__(self):
        self.index = []  # Initialize an empty index
        self.text_corpus = []
        self.name = "Base"

    def indexing(self, documents):
        """
        Index a list of documents.

        Args:
            documents (list): A list of strings to be indexed.
        """
        self.text_corpus = documents

    def get_top_k_documents(self, query, k=10, scores=False):
        """
        Retrieve the top k documents for a given query.

        Args:
            query (str): The query text.
            k (int): The number of top documents to retrieve.
            scores (bool): Whether to return scores along with documents.

        Returns:
            list: A list of top k documents (and scores if scores=True).
        """
        # Implement retrieval logic here
        # You can override this method in derived classes

        # For demonstration purposes, return the first k documents as top documents
        if scores:
            return self.index[:k], [1.0] * k  # Dummy scores
        else:
            return self.index[:k]

    def predict(self, query, text):
        """
        Predict a score for a given query and text.

        Args:
            query (str): The query text.
            text (str): The text to calculate the score for.

        Returns:
            float: The calculated score for the query and text.
        """
        # Implement scoring logic here
        # You can override this method in derived classes

        # For demonstration purposes, return a dummy score
        return 1.0  # Dummy score


class LargeDPROpenAIRetriever(BaseRetriever):
    def __init__(self, openai_api_key, index_name, model_name="text-embedding-ada-002"):
        print("Setting up Large DPR")
        self.openai_client = OpenAI(
            api_key=openai_api_key,
        )
        self.text_corpus = []
        self.questions = []
        self.name = "OpenAI-DPR"
        self.model_name = model_name
        self.embedding_file_path = os.path.join(os.environ['HOME'], 'code', 'retriever', 'data', index_name +"_embeddings" + ".json")
        self.load_embeddings()
        print("Done Initializing")

    def load_embeddings(self):
        """
        Load embeddings from the file if it exists.
        """
        if os.path.exists(self.embedding_file_path):
            print("Loading embeddings from:", self.embedding_file_path)
            with open(self.embedding_file_path, "r") as file:
                data = json.load(file)

            self.index = data.get("embeddings_text_corpus", [])
            none_count_index = sum(1 for i in self.index if i is None)
            self.index = np.array([np.array(embedding, dtype=np.float32) if embedding is not None else np.zeros((1536,), dtype=np.float32) for embedding in self.index])
            print(f"Index: {self.index.shape}, Replaced NoneType elements: {none_count_index}")

            questions = data.get("embeddings_questions", [])
            self.embeddings_questions = []
            none_count_questions = 0
            for question_set in questions:
                none_count_questions += sum(1 for i in question_set if i is None)
                embedded_set = np.array([np.array(embedding, dtype=np.float32) if embedding is not None else np.zeros((1536,), dtype=np.float32) for embedding in question_set])
                self.embeddings_questions.append(embedded_set)
            print(f"Embeddings Questions: {len(self.embeddings_questions)}, Replaced NoneType elements: {none_count_questions}")

        else:
            self.index = []
            self.embeddings_questions = []
            print("Embeddings file not found. Skipping loading embeddings.")

    def save_embeddings(self):
        """
        Save embeddings to a file.
        """
        print("Saving embeddings to:", self.embedding_file_path)
        data = {"embeddings_text_corpus": self.index, "embeddings_questions": self.embeddings_questions}
        with open(self.embedding_file_path, "w") as file:
            json.dump(data, file, indent=2)

    def indexing(self, text_corpus, questions):
        """
        Index a list of documents along with corresponding questions.

        Args:
            text_corpus (list): A list of strings to be indexed.
            questions (list): A list of lists containing questions corresponding to text_corpus.
        """
        print("Starting Indexing")
        indexing_time_start = time.time()

        self.text_corpus = text_corpus

        if len(self.index) == 0 or len(self.embeddings_questions) == 0:
            print("Indexing ...")
            # Embed and index each document only if embeddings are not already loaded
            self.questions = questions
            self.index = []

            # Embed and index each document
            self.index.extend(self.parallel_embed_documents(self.text_corpus))
            for question_set in self.questions:
                self.embeddings_questions.append(self.parallel_embed_documents(question_set))
            # Save the index to a file
            self.save_embeddings()
            self.load_embeddings()
        else:
            print("Embeddings already loaded. Skipping indexing.")

        end_time_indexing = time.time()
        print(f"Done Indexing. Took {end_time_indexing - indexing_time_start} seconds")

    def get_top_k_documents(self, query, k=10, scores=False):
        """
        Retrieve the top k documents for a given query using local embeddings.

        Args:
            query (str): The query embedding.
            k (int): The number of top documents to retrieve.
            scores (bool): Whether to return scores along with documents.

        Returns:
            list: A list of top k documents (and scores if scores=True).
        """
        try:
            # Embed the query
            query_embedding = np.array(query)
            query_embedding = query_embedding.reshape(1, -1)

            # Calculate similarity scores between the query and all documents
            # similarity_scores = [self.calculate_cosine_similarity(query_embedding, doc_embedding) for doc_embedding in self.index]
            similarity_scores = cosine_similarity(query_embedding, self.index)

            # Get indices of top k documents
            top_k_indices = np.argsort(similarity_scores[0])[-k:][::-1]

            # Retrieve the corresponding documents from the text_corpus
            top_k_documents = [self.text_corpus[idx] for idx in top_k_indices]

            if scores:
                document_scores = [similarity_scores[0][idx] for idx in top_k_indices]
                return top_k_documents, document_scores
            else:
                return top_k_documents

        except Exception as e:
            print(f"Local Retrieval Error: {e}")
            raise

    def parallel_embed_documents(self, documents):
        documents_with_indices = list(enumerate(documents))  # Add indices to documents
        with concurrent.futures.ThreadPoolExecutor() as executor:
            embeddings_with_indices = list(tqdm(executor.map(self.embed_text_with_index, documents_with_indices),
                                                total=len(documents_with_indices),
                                                desc="Embedding documents",
                                                unit="document",
                                                file=sys.stdout))
        # Sort the embeddings based on the original document indices
        embeddings_with_indices.sort(key=lambda x: x[0])
        embeddings = [embedding for _, embedding in embeddings_with_indices]
        return embeddings

    def embed_text_with_index(self, document_with_index):
        index, document = document_with_index
        embedding = self.embed_text(document)
        return index, embedding

    def embed_text(self, text):
        """
        Embed text using OpenAI's text embedding model.

        Args:
            text (str): The text to be embedded.

        Returns:
            list: A list of floats representing the embedding.
        """
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.model_name
            )
            return response.data[0].embedding
        except openai.RateLimitError as e:
            # Handle rate limit error gracefully
            print(f"Rate limit exceeded. Waiting for 20 seconds before retrying.")
            time.sleep(20)  # You may adjust the sleep duration
            return self.embed_text(text)
        except Exception as e:
            # Handle other exceptions if needed
            print(f"An error occurred with OpenAI API: {e}")
            return None  # or handle in a way that suits your application

    def predict(self, query, text):
        """
        Predict a score for a given query and text.

        Args:
            query (str): The query text.
            text (str): The text to calculate the score for.

        Returns:
            float: The calculated score for the query and text.
        """
        # Embed both the query and text
        query_embedding = self.embed_text(query)
        text_embedding = self.embed_text(text)

        # Calculate the cosine similarity between query and text embeddings
        similarity_score = self.calculate_cosine_similarity(query_embedding, text_embedding)

        return similarity_score

    @staticmethod
    def calculate_cosine_similarity(embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1 (list): The first embedding.
            embedding2 (list): The second embedding.

        Returns:
            float: Cosine similarity score between the two embeddings.
        """
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        print("1")
        # Calculate magnitudes
        magnitude1 = sum(a ** 2 for a in embedding1) ** 0.5
        print("2")
        magnitude2 = sum(a ** 2 for a in embedding2) ** 0.5
        print("3")
        # Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        print("4")
        return similarity

### Load the index

# Original Index
print("Loading Original Index")
original_index_name = index_name.split("_short")[0].split("_long")[0]

original_index_path = os.path.join(os.environ['HOME'], 'code', 'data_augmentation', 'qa_dataset', original_index_name + ".json")

print("Original Index Path:", original_index_path)

with open(original_index_path, 'r') as file:
    data = json.load(file)

text_corpus = [item["context"] for item in data]
questions = [item["questions"] for item in data]

# Load the retriever
retriever = LargeDPROpenAIRetriever(openai_api_key="None", index_name=original_index_name)

retriever.indexing(text_corpus, questions)

questions_embeddings = retriever.embeddings_questions

### Retrieve top-k contexts for each question_qac

def find_string_index(questions, target_string):
    for outer_index, sub_list in enumerate(questions):
        for inner_index, string_in_sublist in enumerate(sub_list):
            if target_string == string_in_sublist:
                return outer_index, inner_index
    return -1, -1  # Return -1 for both indices if the string is not found

start_time = time.time()

index_to_drop = []

for i, question_set in tqdm(enumerate(questions_embeddings), total=len(questions_embeddings), desc="Processing questions", file=sys.stdout):
    for j, question_embedding in enumerate(question_set):
        # Check if the retrieved document is in the top 1000 documents
        tousand_most_similar_documents = retriever.get_top_k_documents(question_embedding, k=1000, scores=False)
        if text_corpus[i] in tousand_most_similar_documents:
            pass
        else: 
            index_to_drop.append((i,j))

print("Done Retrieving Contexts took: ", time.time() - start_time_script, " seconds")

print("Filtering out questions that do not have the original context in the top 1000 documents")
print("")
print(f"Number of removed questions: {len(index_to_drop)} | Percentage: {len(index_to_drop)/len(questions_embeddings)*100}%")
print("")

### Filter out the questions that do not have the original context in the top 1000 documents

filtered_questions = []
filtered_contexts = []

for i, question_set in enumerate(tqdm(questions, desc="Filtering", total=len(questions), file=sys.stdout)):
    filtered_question_set = [question_set[j] for j in range(len(question_set)) if (i, j) not in index_to_drop]
    if len(filtered_question_set) != 0:
        filtered_questions.append(filtered_question_set)
        filtered_contexts.append(text_corpus[i])

### Save the qac_extended json
save_path = os.path.join(os.environ['HOME'], 'code', 'data_augmentation', 'qa_dataset', index_name + "Filtered.json")

print("Saving to:", save_path)

with open(save_path, 'w') as file:
    json.dump({"questions": filtered_questions, "context": filtered_contexts}, file, indent=2)