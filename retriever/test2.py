print("Starting Script")
import time

start_time_script = time.time()

import csv
import spacy
from rank_bm25 import BM25Okapi

from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer

from openai import OpenAI
import openai
import pinecone

import os
import json
import sys

import itertools
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


import numpy as np
from tqdm import tqdm


print("Imported all Dependencies")

import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Your script description")

def parse_k_value(value):
    try:
        # Try to parse as an integer
        return int(value)
    except ValueError:
        tmp_value = str(value).replace("'", "")
        print(f"Got for k: {tmp_value}")
        # If it's not an integer, try to parse as a list of integers
        return [int(x) for x in tmp_value.split(',')]

# Add command line arguments
parser.add_argument("--language", type=str, help="Specify the language")
parser.add_argument("--indexName", type=str, help="Specify the index name")
parser.add_argument("--retriever", type=str, help="Specify the retriever")
parser.add_argument("--k", type=parse_k_value, help="Specify the k value")
parser.add_argument("--k_outer", type=int, help="Specify the k_outer value")
parser.add_argument("--ensemble", type=bool, help="Use ensemble")

# Parse the command line arguments
args = parser.parse_args()

# Access the values of the arguments
language = args.language
index_name = args.indexName
retriever = args.retriever
k = args.k
if isinstance(k, list) and len(k) == 1:
    k = k[0]
k_outer = args.k_outer
ensemble = args.ensemble

# Check if required arguments are provided
if not index_name:
    print("No value provided for --indexName argument.")
    sys.exit(1)

if not language:
    print("No value provided for --language argument.")
    sys.exit(1)

# Example usage:
print(f"Language: {language}")
print(f"Index Name: {index_name}")
print(f"Retriever: {retriever}")
print(f"k: {k}")
print(f"k_outer: {k_outer}")
print(f"Ensemble: {ensemble}")


if language == "en":
    print("Load English Spacy ...")
    nlp = spacy.load("en_core_web_sm")  # Load the English SpaCy model
    print("Loaded English Spacy")
elif language == "de":
    print("Load German Spacy ..")
    nlp = spacy.load("de_core_news_sm")
    print("Loaded German Spacy")


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



class BM25Retriever(BaseRetriever):
    def __init__(self, k=10):
        print("Setting up BM25")
        super().__init__()
        self.name = "BM25"
        self.k = k
        print("Done")

    def indexing(self, documents):
        """
        Index a list of documents using BM25Okapi.

        Args:
            documents (list): A list of strings to be indexed.
        """
        start_time_indexing = time.time()
        super().indexing(documents)
        # tokenized_corpus = [self.preprocess(doc) for doc in documents]
        tokenized_corpus = self.batch_preprocess(documents)
        self.index = BM25Okapi(tokenized_corpus)
        print(f"Indexing took {time.time() - start_time_indexing} seconds")
    
    def batch_preprocess(self, texts):
        print("Starting Batch Processing")
        docs = list(nlp.pipe(texts, disable=["parser", "ner"], batch_size=1000))
        print("Done Splitting to Docs")
        
        processed_texts = []
        total_texts = len(texts)

        # Use tqdm for a progress bar
        for doc in tqdm(docs, total=total_texts, desc="Preprocessing", file=sys.stdout):
            tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.is_stop]
            processed_texts.append(tokens)
        print("Done Tokenization")

        return processed_texts

    def preprocess(self, text):
        # Tokenize, stem, and remove stop words using SpaCy
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.is_stop]
        return tokens

    def get_top_k_documents(self, query, k=None, scores=False):
        """
        Retrieve the top k documents for a given query using BM25Okapi.

        Args:
            query (str): The query text.
            k (int): The number of top documents to retrieve.
            scores (bool): Whether to return scores along with documents.

        Returns:
            list: A list of top k documents (and scores if scores=True).
        """
        if k is None:
            k = self.k
        if not self.index:
            raise ValueError("BM25Okapi index has not been created. Please call indexing() first.")

        tokenized_query = self.preprocess(query)
        doc_scores = self.index.get_scores(tokenized_query)

        # Sort documents by score and get the top k documents
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k]
        top_documents = [self.text_corpus[i] for i in top_indices]

        if scores:
            top_scores = [doc_scores[i] for i in top_indices]
            return top_documents, top_scores
        else:
            return top_documents

    def predict(self, query, passage):
        # Tokenize, stem, and remove stop words from the passage and query, then calculate BM25 score
        passage_tokens = self.preprocess(passage)
        query_tokens = self.preprocess(query)
        bm25 = BM25Okapi([passage_tokens])
        return bm25.get_scores(query_tokens)

class CERetriever(BaseRetriever):
    def __init__(self, k=10, model_name='cross-encoder/ms-marco-MiniLM-L-2-v2'):
        print("Setting up CE")
        super().__init__()
        self.ce = CrossEncoder(model_name, max_length=512)
        self.name = "Cross-Encoder"
        self.k = k
        print("Done")

    def get_top_k_documents(self, query, k=None, scores=False):
        """
        Retrieve the top k documents for a given query.

        Args:
            query (str): The query text.
            k (int): The number of top documents to retrieve.
            scores (bool): Whether to return scores along with documents.

        Returns:
            list: A list of top k documents (and scores if scores=True).
        """
        if k is None:
            k = self.k

        if not self.text_corpus:
            raise ValueError("The text corpus has not been set. Please call indexing() first.")

        query_passage_pairs = [(query, passage) for passage in self.text_corpus]

        # Calculate cross encoder scores between the query and documents
        scores_ce = self.ce.predict(query_passage_pairs)

        # Sort the pairs based on scores in descending order
        sorted_pairs = sorted(zip(query_passage_pairs, scores_ce), key=lambda x: x[1], reverse=True)

        # Unzip the sorted pairs to get the sorted documents and scores
        sorted_documents, sorted_scores = zip(*sorted_pairs)

        sorted_passages = [pair[1] for pair in sorted_documents]

        # Extract the top-k documents and scores if requested
        if scores:
            return sorted_passages[:k], sorted_scores[:k]
        else:
            return sorted_passages[:k]

class LargeDPROpenAIRetriever(BaseRetriever):
    def __init__(self, pinecone_api_key, openai_api_key, index_name, model_name="text-embedding-ada-002"):
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
                self.embeddings_questions = data.get("embeddings_questions", [])
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
            json.dump(data, file, indent=2)  # Added indentation for better readability

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

        if not self.index or not self.embeddings_questions:
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
            query_embedding = query

            # Calculate similarity scores between the query and all documents
            similarity_scores = [self.calculate_cosine_similarity(query_embedding, doc_embedding) for doc_embedding in self.index]

            # Get indices of top k documents
            top_k_indices = np.argsort(similarity_scores)[-k:][::-1]

            # Retrieve the corresponding documents from the text_corpus
            top_k_documents = [self.text_corpus[idx] for idx in top_k_indices]

            if scores:
                document_scores = [similarity_scores[idx] for idx in top_k_indices]
                return top_k_documents, document_scores
            else:
                return top_k_documents

        except Exception as e:
            print(f"Local Retrieval Error: {e}")
            raise

    def parallel_embed_documents(self, documents):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            embeddings = list(tqdm(executor.map(self.embed_text, documents),
                                   total=len(documents),
                                   desc="Embedding documents",
                                   unit="document",
                                   file=sys.stdout))
        
        # Assuming self.index is the list where you want to store embeddings
        return embeddings
        # self.index.extend(embeddings)

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


    # def upload_index_to_pinecone(self):
    #     """
    #     Upload the index (embeddings) to Pinecone.
    #     """
    #     # Delete the existing index if it exists
    #     try:
    #         pinecone.delete_index(self.pinecone_index_name)
    #         print(f"Deleted Pinecone index '{self.pinecone_index_name}'.")
    #     except pinecone.api_index.IndexNotFoundException:
    #         pass

    #     # Create a new index
    #     pinecone.create_index(self.pinecone_index_name, dimension=1536)
    #     print(f"Created Pinecone index '{self.pinecone_index_name}'.")

    #     embeddings = self.index  # Assuming self.index contains the document embeddings
    #     document_ids = list(range(len(embeddings)))

    #     # # Convert embeddings to Pinecone's Item format
    #     # items = [(str(id), embedding) for id, embedding in zip(document_ids, embeddings)]

    #     # # Upsert the items into Pinecone
    #     # for item in items:
    #     #     self.pinecone_index.upsert(item)
    #     def chunks(iterable, batch_size=100):
    #         """A helper function to break an iterable into chunks of size batch_size."""
    #         it = iter(iterable)
    #         chunk = list(itertools.islice(it, batch_size))
    #         while chunk:
    #             yield chunk
    #             chunk = list(itertools.islice(it, batch_size))

    #     # Convert embeddings to Pinecone's Item format
    #     items = [(str(id), embedding) for id, embedding in zip(document_ids, embeddings)]

    #     # Upsert data with 100 vectors per upsert request
    #     total_batches = len(items) // 100 + (len(items) % 100 > 0)

    #     for items_chunk in tqdm(chunks(items, batch_size=100), total=total_batches, desc="Upserting items to pinecone", unit="batch"):
    #         self.pinecone_index.upsert(items_chunk)

    # def get_top_k_documents(self, query, k=10, scores=False):
    #     """
    #     Retrieve the top k documents for a given query.

    #     Args:
    #         query (str): The query text.
    #         k (int): The number of top documents to retrieve.
    #         scores (bool): Whether to return scores along with documents.

    #     Returns:
    #         list: A list of top k documents (and scores if scores=True).
    #     """
    #     try:
    #         # Embed the query
    #         query_embedding = self.embed_text(query)

    #         # Perform similarity search in Pinecone
    #         results = pinecone.Index(self.pinecone_index_name).query(query_embedding, top_k=k, include_metadata=True)

    #         # Extract document IDs, scores, and metadata
    #         document_ids = [result.id for result in results["matches"]]
    #         document_scores = [result.score for result in results["matches"]]
    #         metadata = [result.metadata for result in results["matches"]]

    #         # Retrieve the corresponding documents from the text_corpus
    #         top_k_documents = [self.text_corpus[int(id)] for id in document_ids]

    #         if scores:
    #             return top_k_documents, document_scores
    #         else:
    #             return top_k_documents
        
    #     except Exception as e:
    #         print(f"Pinecone Error: {e}")
    #         raise

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
        # Calculate magnitudes
        magnitude1 = sum(a ** 2 for a in embedding1) ** 0.5
        magnitude2 = sum(a ** 2 for a in embedding2) ** 0.5
        # Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        return similarity

class RetrieverEnsemble:
    def __init__(self, retrievers, text_corpus):
        print("Setting up Ensemble")
        self.retrievers = retrievers
        self.text_corpus = text_corpus
        self.name = "Ensemble"
        self.retrievers[0].indexing(text_corpus)
        print("Done")

    def get_top_k_documents(self, query, k=None, scores=True):
        # if k is None:
        #     # Use the default k value of the first retriever in the list
        #     k = self.retrievers[0].k

        retrieved_passages = self.text_corpus
        for i in range(len(self.retrievers)):
            if i != 0:
                self.retrievers[i].indexing(retrieved_passages)
            # results = self.retrievers[i].get_top_k_documents(query,k,scores=True)
            results = self.retrievers[i].get_top_k_documents(query,scores=True)
            if i == len(self.retrievers):
                if scores:
                    return results
                else:
                    return results[0]
            else:
                retrieved_passages = results[0]

        return results

# Get Index

index_file_path = os.path.join(os.environ['HOME'], 'code', 'data_augmentation', 'qa_dataset', index_name + ".json")

print("Start Loading Data ...")

with open(index_file_path, 'r') as file:
    data = json.load(file)

text_corpus = [item["context"] for item in data]
questions = [item["questions"] for item in data]

# Comment this lines till total questions out for full benchmark run
total_questions = sum(len(question_set) for question_set in questions)

print(f"Number Text Corpus Entries: {len(text_corpus)} | Number Questions: {total_questions}")

text_corpus = text_corpus[:10]
questions = questions[:5]

total_questions = sum(len(question_set) for question_set in questions)

print(f"Number Text Corpus Entries: {len(text_corpus)} | Number Questions: {total_questions}")

print(f"Loaded Data. Everything up here took {time.time() - start_time_script} seconds") #takes 7.590401887893677 min

# Calculate hit and MRR function:
from multiprocessing import Pool

def calculate_hit_ratio_and_mrr_parallel(args):
    retriever, text_corpus, questions, k, ensemble = args
    total_hits = 0
    reciprocal_ranks = 0
    total_questions = 0

    for i, question_set in tqdm(enumerate(questions), total=len(questions), desc=f"Processing questions for k: {k} and #questions: {len(questions)}", file=sys.stdout):
        for question in question_set:
            context = text_corpus[i]
            try:
                if ensemble:
                    top_k_documents, document_scores = retriever.get_top_k_documents(question, scores=True)
                    top_k_documents = top_k_documents[:k]
                    document_scores = top_k_documents[:k]

                else:
                    results = retriever.get_top_k_documents(question, k=k, scores=True)
                    top_k_documents, document_scores = results

                if context in top_k_documents:
                    total_hits += 1
                    rank = top_k_documents.index(context) + 1
                    reciprocal_ranks += 1 / rank

                total_questions += 1

            except:
                pass

    try:
        hit_ratio = total_hits / total_questions
        mrr = reciprocal_ranks / total_questions

        return k, hit_ratio, mrr, total_questions
    except:
        return None

def calculate_hit_ratio_and_mrr_parallel_wrapper(retriever, text_corpus, questions, k_values, ensemble, multithreading=False):
    results = []
    num_cores = os.cpu_count()
    print(f"Number Cores: {num_cores}")

    start_time_wrapper = time.time()

    questions_parts = [questions[i::num_cores] for i in range(num_cores)]

    print(f"Question Parts Len: {len(questions_parts)}")

    if ensemble == False:
        with Pool(len(questions_parts)) as pool:
            results = pool.map(
                calculate_hit_ratio_and_mrr_parallel,
                [(retriever, text_corpus, part, k_values, ensemble) for part in questions_parts] # split questions into # Pools many parts
            )
            results = [i for i in results if i is not None]
            array_result = np.array(results)
            weights = array_result[:, 3]
            results = np.average(array_result[:, :3], axis=0, weights=weights)
    elif multithreading == True:
        print("Start Multithreading")
        with ThreadPoolExecutor(len(questions_parts)) as executor:
            results = list(executor.map(
                calculate_hit_ratio_and_mrr_parallel,
                [(retriever, text_corpus, chunk, k_values, ensemble) for chunk in questions_parts]
            ))
            # results = [i[:3] for i in results]

            # filter out None from results
            results = [i for i in results if i is not None]
            print(f"Done Multithreading. Results: {results}")

            # Accumulate the results after parallel execution
            accumulated_results = np.array(results)

            # Process the accumulated results if needed
            # For example, calculate the weighted average as in the Pool case
            weights = accumulated_results[:, 3]
            final_result = np.average(accumulated_results[:, :3], axis=0, weights=weights)
            results = final_result
    else:
        results = calculate_hit_ratio_and_mrr_parallel((retriever, text_corpus, questions, k_values, ensemble))
        print("Results: ", results)
        results = results[:3]

    elapsed_time_wrapper = time.time() - start_time_wrapper

    print(f"Elapsed Time Wrapper: {elapsed_time_wrapper:.4f} seconds")

    print(f"Results: {results}")
    return results

# Indices to evaluate

if retriever == "BM25":
    # 1. BM25

    start_time_setup = time.time()
    # Create BM25 Retriever with indexing based on text corpus
    bm25_retriever = BM25Retriever(k=25)
    bm25_retriever.indexing(text_corpus)
    
    elapsed_time_setup = time.time() - start_time_setup

    print(f"Setup of Retriever took: {elapsed_time_setup} seconds")

    # Example usage:
    start_time_bench = time.time()  # Record the start time

    print("Starting Benchmark BM25 ...")
    k_values = k
    unpacked_list = calculate_hit_ratio_and_mrr_parallel_wrapper(bm25_retriever, text_corpus, questions, k_values,False)
    k, hit_ratio, mrr = unpacked_list 

    # Write the results to a CSV file
    output_file = f'./benchmarks/bm25/{language}/{index_name}/hit_ratio_mrr_results_BM25_{k}.csv'
    output_directory = os.path.dirname(output_file)
    os.makedirs(output_directory, exist_ok=True)
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Retriever','k', 'HR', 'MRR'])  # Write header
        csv_writer.writerow(["BM25", int(k), hit_ratio, mrr])

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time_bench  # Calculate the elapsed time

    print(f'Results saved to {output_file}')
    print(f'Total time taken: {elapsed_time} seconds')

elif retriever == "BM25+CE":
    # 2. BM25 + CE
    print("Benchmarking Ensemble")
    # Create Ensemble Retriever 
    k_bm25 = k  # Different k values for BM25
    k_ce = k_outer     # Different k values for CE

    ensemble_results = {}  # To store results for different configurations

    retriever1 = BM25Retriever(k=k)
    if language == "de":
        model_name = "cross-encoder/msmarco-MiniLM-L6-en-de-v1"
        retriever2 = CERetriever(k=k_ce,model_name=model_name)
    else:
        retriever2 = CERetriever(k=k_ce)

    retrievers = [retriever1, retriever2]

    bm25_and_ce_retriever_ensemble = RetrieverEnsemble(retrievers, text_corpus)

    k_config = f"BM25-k{k_bm25}-CE-k{k_ce}"

    # Example usage for the ensemble (BM25 + CE)
    k_value, hit_ratio, mrr_value = calculate_hit_ratio_and_mrr_parallel_wrapper(bm25_and_ce_retriever_ensemble, text_corpus, questions, k_ce, ensemble=True)

    # Store the results in a dictionary
    ensemble_results[k_config] = (hit_ratio, mrr_value)

    print(f"Done Benchmark for Ensemble: k_bm25: {k_bm25} and k_ce: {k_ce}")

    # Write the results to CSV files for different configurations
    for k_config, (hit_ratios, mrr_values) in ensemble_results.items():
        output_file = f'./benchmarks/bm25_ce/{language}/{index_name}/hit_ratio_mrr_results_{k_config}.csv'
        output_directory = os.path.dirname(output_file)
        os.makedirs(output_directory, exist_ok=True)
        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Retriever', 'k', 'HR', 'MRR'])  # Write header
            csv_writer.writerow([k_config, k_ce, hit_ratio, mrr_value])

        print(f'{k_config} Results saved to {output_file}')

elif retriever == "DPR":
    # 3. Large DPR

    # Create DPR Retriever with indexing based on text corpus
    print("Benchmarking Large DPR")
    # # Initialize the retriever with your Pinecone API key
    pinecone_api_key = "e64631a4-5aae-471b-97ba-a44f0f953bf8"
    openai_api_key = "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"

    large_dpr_retriever = LargeDPROpenAIRetriever(pinecone_api_key, openai_api_key, index_name)

    large_dpr_retriever.indexing(text_corpus, questions)

    embeddings_questions = large_dpr_retriever.embeddings_questions

    for k_iter in k:
        results = []
        k_value, hit_ratio, mrr_value = calculate_hit_ratio_and_mrr_parallel_wrapper(large_dpr_retriever, text_corpus, embeddings_questions, k_iter, ensemble=True, multithreading=False)
        results.append([k_value, hit_ratio, mrr_value])

        print(f'Done Benchmarking for k={k_iter}')

        # Write the results to a CSV file for each k iteration
        output_file = f'./benchmarks/dpr/{language}/{index_name}/hit_ratio_mrr_results_{k_iter}.csv'
        output_directory = os.path.dirname(output_file)
        os.makedirs(output_directory, exist_ok=True)
        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Retriever', 'k', 'HR', 'MRR'])  # Write header
            for row in results:
                csv_writer.writerow(["largeDPR", row[0], row[1], row[2]])

        print(f'Results saved to {output_file}')

print(f"Script runtime: {time.time() - start_time_script}")