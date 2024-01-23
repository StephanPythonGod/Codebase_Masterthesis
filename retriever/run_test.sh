#!/bin/bash

#SBATCH --output=output_test5.txt
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=40gb
#SBATCH --job-name=retriever-benchmarking-test

language="en"
indexName="IndexEnglishAll"

# Load the modules
echo "Starting ..."

module load devel/cuda/11.8
module load devel/python/3.8.6_gnu_10.2
# module load devel/python/3.8.6_intel_19.1
# module load devel/miniconda/4.9.2

echo "Modules loaded"

# create a virtual environment
if [ ! -d "venv-python3" ]; then
    echo "Creating Venv"
    python3.8 -m venv venv-python3
fi

# activate the venv
. venv-python3/bin/activate

echo "Virutal Env Activated"

pip3.8 install --upgrade pip

pip3.8 --version

pip3.8 install -U setuptools wheel

pip3.8 install spacy transformers rank_bm25 scipy scikit-learn sentence-transformers openai pinecone-client

pip3.8 install --upgrade numpy scipy

python3 -m spacy download en_core_web_trf
python3 -m spacy download de_dep_news_trf

echo "Installed Dependencies"

echo "Running python script with language: $language, indexName: $indexName"

stdbuf -oL -eL python3.8 test.py --language "$language" --indexName "$indexName" > "py_output_test5_${language}_${indexName}.txt" 2>&1
# python3.8 test.py --language "$language" --indexName "$indexName" > "py_output_test_${language}_${indexName}.txt" 2>&1
# python3 test.py --language "$language" --indexName "$indexName"

echo "Script ran through"
# Deactivate the virtual environment
deactivate