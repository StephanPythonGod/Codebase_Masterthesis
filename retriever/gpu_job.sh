#!/bin/bash

#SBATCH --output=$1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=40gb
#SBATCH --job-name=gpu-job-benchmarking

module load devel/cuda/11.8
module load devel/python/3.8.6_gnu_10.2

echo "Modules loaded"

# create a virtual environment
if [ ! -d "venv-python3" ]; then
    echo "Creating Venv"
    python3.8 -m venv venv-python3
else
    echo "Venv Exists"
fi

# activate the venv
. venv-python3/bin/activate

echo "Virtual Env Activated"

pip3.8 install --upgrade pip

pip3.8 --version

pip3.8 install -U setuptools wheel

pip3.8 install spacy transformers rank_bm25 scipy scikit-learn sentence-transformers openai pinecone-client tqdm

pip3.8 install --upgrade numpy scipy

python3 -m spacy download en_core_web_sm
python3 -m spacy download de_core_news_sm

echo "Installed Dependencies"

echo "Running python script with retriever: $retriever, language: $language, indexName: $indexName, k: $k, k_outer: $k_outer, ensemble: $ensemble, output file: $output_file"

python3.8 -u ./test.py \
  --language "$language" \
  --indexName "$indexName" \
  --retriever "$retriever" \
  --k "$k" \
  --k_outer "$k_outer" \
  --ensemble "$ensemble" > "$output_file" 2>&1

echo "Python script completed"