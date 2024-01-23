#!/bin/bash

#SBATCH --output=output_en_all.txt
#SBATCH --partition=gpu_4
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40gb
#SBATCH --job-name=retriever-benchmarking-en-all

language="en"
indexName="IndexEnglishAll"

# Load the modules
echo "Starting ..."

module load devel/cuda/11.8
module load devel/python/3.8.6_gnu_10.2

echo "Modules loaded"

# create a virtual environment
if [ ! -d "venv-python3" ]; then
    echo "Creating Venv"
    python -m venv venv-python3
fi

# activate the venv
. venv-python3/bin/activate


echo "Virutal Env Activated"

pip install --upgrade pip

# dependencies
pip install transformers rank_bm25 scipy scikit-learn sentence-transformers openai pinecone-client

# change language for BM25 preprocessing
pip install -U pip setuptools wheel
pip install -U spacy

python3 -m spacy download en_core_web_trf
python3 -m spacy download de_dep_news_trf

#pip install -r requirements_dataaug.txt

echo "Installed Dependencies"

python3 retriever_benchmarking_en.py --language "$language" --indexName "$indexName" > py_output_$language_$indexName.txt 2>&1

echo "Script ran through"
# Deactivate the virtual environment
deactivate