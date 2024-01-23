#!/bin/bash

#SBATCH --output=output_context_generation.txt
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=40gb
#SBATCH --job-name=context

language="de"
# language="en"

# indexName="IndexGermanAll"
# indexName="IndexGermanAll_short"
# indexName="IndexEnglishAll_short"
# indexName="translated_IndexEnglishAll"
# indexName="translated_IndexGermanAll"
# indexName="translated_IndexGermanAll_short"
# indexName="translated_IndexEnglishAll_short"

# indexName="IndexEnglishAllFiltered_long"
# indexName="IndexGermanAllFiltered_long"
indexName="translated_IndexEnglishAllFiltered_long"
# indexName="translated_IndexGermanAllFiltered_long"

# filterOut="True"
filterOut="False"

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
pip install openai tqdm numpy scikit-learn

echo "Installed Dependencies"

python3 -u ./context_generation.py \
  --language "$language" \
  --filterOut "$filterOut" \
  --indexName "$indexName" > "py_output_context_generation_${indexName}.txt" 2>&1


echo "Script ran through"