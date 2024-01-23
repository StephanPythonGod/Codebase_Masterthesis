import csv
import spacy
from rank_bm25 import BM25Okapi

from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer

import openai
import pinecone

import os
import json
import sys

# Check if the "--language" argument was provided
if "--language" in sys.argv:
    language_index = sys.argv.index("--language")  # Find the index of "--language" in sys.argv
    if language_index + 1 < len(sys.argv):
        language = sys.argv[language_index + 1]
    else:
        print("No value provided for --language argument.")
        sys.exit(1)
else:
    print("No --language argument provided.")
    sys.exit(1)

if language == "en":
    print("Load English Spacy ...")
    nlp = spacy.load("en_core_web_trf")  # Load the English SpaCy model
    print("Loaded English Spacy")
elif language == "de":
    print("Load German Spacy ..")
    nlp = spacy.load("de_dep_news_trf")
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
        super().__init__()
        self.name = "BM25"
        self.k = k

    def indexing(self, documents):
        """
        Index a list of documents using BM25Okapi.

        Args:
            documents (list): A list of strings to be indexed.
        """
        super().indexing(documents)
        tokenized_corpus = [self.preprocess(doc) for doc in documents]
        self.index = BM25Okapi(tokenized_corpus)

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
        super().__init__()
        self.ce = CrossEncoder(model_name, max_length=512)
        self.name = "Cross-Encoder"
        self.k = k

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
    def __init__(self, pinecone_api_key, openai_api_key, model_name="text-embedding-ada-002"):
        openai.api_key = openai_api_key
        self.text_corpus = []
        self.name = "OpenAI-DPR"
        self.model_name = model_name
        pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
        self.pinecone_index_name = "masterthesis"
        self.pinecone_index = pinecone.Index(self.pinecone_index_name)

    def indexing(self, documents):
        """
        Index a list of documents.

        Args:
            documents (list): A list of strings to be indexed.
        """
        self.text_corpus = documents
        self.index = []

        # Embed and index each document
        for document in documents:
            embedding = self.embed_text(document)
            self.index.append(embedding)

        # Upload the index to Pinecone
        self.upload_index_to_pinecone()

    def embed_text(self, text):
        """
        Embed text using OpenAI's text embedding model.

        Args:
            text (str): The text to be embedded.

        Returns:
            list: A list of floats representing the embedding.
        """
        response = openai.Embedding.create(
            input=text,
            model=self.model_name
        )
        return response["data"][0]["embedding"]

    def upload_index_to_pinecone(self):
        """
        Upload the index (embeddings) to Pinecone.
        """
        # Delete the existing index if it exists
        try:
            pinecone.delete_index(self.pinecone_index_name)
            print(f"Deleted Pinecone index '{self.pinecone_index_name}'.")
        except pinecone.api_index.IndexNotFoundException:
            pass

        # Create a new index
        pinecone.create_index(self.pinecone_index_name, dimension=1536)
        print(f"Created Pinecone index '{self.pinecone_index_name}'.")

        embeddings = self.index  # Assuming self.index contains the document embeddings
        document_ids = list(range(len(embeddings)))

        # Convert embeddings to Pinecone's Item format
        items = [(str(id), embedding) for id, embedding in zip(document_ids, embeddings)]

        # Upsert the items into Pinecone
        self.pinecone_index.upsert(items)

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
        # Embed the query
        query_embedding = self.embed_text(query)

        # Perform similarity search in Pinecone
        results = pinecone.Index(self.pinecone_index_name).query(query_embedding, top_k=k, include_metadata=True)

        # Extract document IDs, scores, and metadata
        document_ids = [result.id for result in results["matches"]]
        document_scores = [result.score for result in results["matches"]]
        metadata = [result.metadata for result in results["matches"]]

        # Retrieve the corresponding documents from the text_corpus
        top_k_documents = [self.text_corpus[int(id)] for id in document_ids]

        if scores:
            return top_k_documents, document_scores
        else:
            return top_k_documents

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
        self.retrievers = retrievers
        self.text_corpus = text_corpus

    def get_top_k_documents(self, query, k=None, scores=True):
        # if k is None:
        #     # Use the default k value of the first retriever in the list
        #     k = self.retrievers[0].k

        retrieved_passages = self.text_corpus
        for i in range(len(self.retrievers)):
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

# Check if the "--language" argument was provided
if "--indexName" in sys.argv:
    index_name_index = sys.argv.index("--indexName")  # Find the index of "--language" in sys.argv
    if index_name_index + 1 < len(sys.argv):
        index_name = sys.argv[index_name_index + 1]
    else:
        print("No value provided for --indexName argument.")
        sys.exit(1)
else:
    print("No --language argument provided.")
    sys.exit(1)


index_file_path = os.path.join(os.environ['HOME'], 'code', 'data_augmentation', 'qa_dataset', index_name + ".json")

print("Start Loading Data ...")

with open(index_file_path, 'r') as file:
    data = json.load(file)

text_corpus = [item["context"] for item in data]
questions = [item["questions"] for item in data]

print("Loaded Data")

# Calculate hit and MRR function:

def calculate_hit_ratio_and_mrr(retriever_to_evaluate, text_corpus, questions, k_values, ensemble=False):
    hit_ratios = []
    mrr_values = []
    
    for k in k_values:
        total_hits = 0
        reciprocal_ranks = 0
        total_questions = 0

        for i, question_set in enumerate(questions):
            for question in question_set:
                context = text_corpus[i]  # Identify the context corresponding to the question
                if ensemble:
                    top_k_documents, document_scores = retriever_to_evaluate.get_top_k_documents(question, scores=True)
                    top_k_documents = top_k_documents[:k]
                    document_scores = top_k_documents[:k]
                else:
                    results = retriever_to_evaluate.get_top_k_documents(question, k=k, scores=True)
                    top_k_documents, document_scores = results

                print(len(top_k_documents), context in top_k_documents)

                if context in top_k_documents:
                    total_hits += 1
                    rank = top_k_documents.index(context) + 1
                    reciprocal_ranks += 1 / rank
                
                total_questions += 1

        hit_ratio = total_hits / total_questions
        mrr = reciprocal_ranks / total_questions
        
        hit_ratios.append(hit_ratio)
        mrr_values.append(mrr)
    
    return hit_ratios, mrr_values

# Indices to evaluate

# 1. BM25

# Create BM25 Retriever with indexing based on text corpus
print("Setting up BM25")        
bm25_retriever = BM25Retriever(k=25)
bm25_retriever.indexing(text_corpus)
print("BM25 Setup done")

# Example usage:
k_values = [1, 3, 5, 10, 15, 20, 25]  # List of k values you want to evaluate
hit_ratios, mrr_values = calculate_hit_ratio_and_mrr(bm25_retriever, text_corpus, questions, k_values)

# Write the results to a CSV file
output_file = f'./benchmarks/hit_ratio_mrr_results_BM25_{index_name}.csv'
with open(output_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Retriever','k', 'Hit Ratio', 'MRR'])  # Write header
    for k, hit_ratio, mrr in zip(k_values, hit_ratios, mrr_values):
        csv_writer.writerow(["BM25",k, hit_ratio, mrr])

print(f'Results saved to {output_file}')

# 2. BM25 + CE

print("Benchmarking Ensemble")
# Create Ensemble Retriever 
k_values_bm25 = [50, 75, 100]  # Different k values for BM25
k_values_ce = [5, 10, 20]     # Different k values for CE

ensemble_results = {}  # To store results for different configurations

for k_bm25 in k_values_bm25:
    for k_ce in k_values_ce:
        print("Setting up BM25")
        retriever1 = BM25Retriever(k=k_bm25)
        print("Done")
        print("Setting up CE")
        if language == "de":
            model_name = "cross-encoder/msmarco-MiniLM-L6-en-de-v1"
            retriever2 = CERetriever(k=k_ce,model_name=model_name)
        else:
            retriever2 = CERetriever(k=k_ce)
        print("Done")
        retrievers = [retriever1, retriever2]

        print("Setting up Ensemble")
        bm25_and_ce_retriever_ensemble = RetrieverEnsemble(retrievers, text_corpus)
        print("Done")

        k_config = f"BM25-k{k_bm25}-CE-k{k_ce}"

        # Example usage for the ensemble (BM25 + CE)
        hit_ratios, mrr_values = calculate_hit_ratio_and_mrr(bm25_and_ce_retriever_ensemble, text_corpus, questions, k_values, ensemble=True)

        # Store the results in a dictionary
        ensemble_results[k_config] = (hit_ratios, mrr_values)

        print(f"Done Benchmark for Ensemble: k_bm25: {k_bm25} and k_ce: {k_ce}")

# Write the results to CSV files for different configurations
for k_config, (hit_ratios, mrr_values) in ensemble_results.items():
    output_file = f'./benchmarks/hit_ratio_mrr_results_{k_config}_{index_name}.csv'
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Retriever', 'k', 'Hit Ratio', 'MRR'])  # Write header
        for k, hit_ratio, mrr in zip(k_values, hit_ratios, mrr_values):
            csv_writer.writerow([k_config, k, hit_ratio, mrr])

    print(f'{k_config} Results saved to {output_file}')

# 3. Large DPR

# Create DPR Retriever with indexing based on text corpus


print("Benchmarking Large DPR")
# # Initialize the retriever with your Pinecone API key
pinecone_api_key = "e64631a4-5aae-471b-97ba-a44f0f953bf8"
openai_api_key = "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"

print("Setting up Large DPR")
large_dpr_retriever = LargeDPROpenAIRetriever(pinecone_api_key, openai_api_key)
print("Done")

large_dpr_retriever.indexing(text_corpus)

hit_ratios, mrr_values = calculate_hit_ratio_and_mrr(large_dpr_retriever, text_corpus, questions, k_values)

print("Done Benchmarking")

# Write the results to a CSV file
output_file = f'./benchmarks/hit_ratio_mrr_results_largeDPR_{index_name}.csv'
with open(output_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Retriever','k', 'Hit Ratio', 'MRR'])  # Write header
    for k, hit_ratio, mrr in zip(k_values, hit_ratios, mrr_values):
        csv_writer.writerow(["largeDPR",k, hit_ratio, mrr])

print(f'Results saved to {output_file}')