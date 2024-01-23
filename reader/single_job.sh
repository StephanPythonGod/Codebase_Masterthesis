#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=single
#SBATCH --output=$1
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=40gb
#SBATCH --job-name=$2

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

pip install --upgrade pip

# dependencies
pip install pymongo transformers torch google-cloud-storage evaluate openai nltk absl-py rouge_score bert_score tqdm

# english Llama-2-GPTQ
pip install optimum>=1.12.0
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/  # Use cu117 if on CUDA 11.7

echo "Installed Dependencies"

echo "Running python script with reader: $reader, language: $language, indexName: $indexName"

python3.8 -u ./test.py \
  --language "$language" \
  --indexName "$indexName" \
  --dataset "$dataset" \
  --contextLength "$cl" \
  --randomShuffle "$random" \
  --batchSize "$batchSize" \
  --split "$split" \
  --splitIndex "$splitIndex" \
  --reader "$reader" > "$output_file" 2>&1

echo "Python script completed"