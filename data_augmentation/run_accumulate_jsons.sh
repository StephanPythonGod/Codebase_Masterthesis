#!/bin/bash

#SBATCH --output=accumulate_jsons.txt
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=40gb
#SBATCH --job-name=accumulate

# indexName="IndexEnglishAllFiltered_long"
# indexName="IndexGermanAllFiltered_long"
# indexName="translated_IndexEnglishAllFiltered_long"
indexName="translated_IndexGermanAllFiltered_long"

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

echo "Installed Dependencies"

python3 -u ./accumulate_jsons.py \
  --indexName "$indexName" > "py_output_accumulate_jsons_${indexName}.txt" 2>&1

echo "Script ran through"