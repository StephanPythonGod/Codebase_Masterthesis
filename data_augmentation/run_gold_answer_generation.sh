#!/bin/bash

#SBATCH --output=output_gold_answer_generation.txt
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=40gb
#SBATCH --job-name=dataaug

language="de"
# language="en"
indexName="IndexGermanAll"
# indexName="IndexEnglishAll"
# indexName="translated_IndexEnglishAll"
# indexName="translated_IndexGermanAll"

# Load the modules
echo "Starting ..."

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
pip install openai tqdm

echo "Installed Dependencies"

python3 -u ./gold_answer_generation.py \
  --language "$language" \
  --indexName "$indexName" > "py_output_gold_answer_generation.txt" 2>&1


echo "Script ran through"