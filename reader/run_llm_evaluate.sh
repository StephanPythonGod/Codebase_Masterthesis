#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=single
#SBATCH --output=run_llm_evaluate.txt
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=40gb
#SBATCH --job-name=llm_evaluate

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
pip install openai

echo "Installed Dependencies"

echo "Running python script with reader: $reader, language: $language, indexName: $indexName"

python3.8 -u ./re_evaluate_llm.py > "py_out_re_evaluate_llm.txt" 2>&1

echo "Python script completed"