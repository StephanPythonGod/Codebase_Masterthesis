#!/bin/bash

#SBATCH --output=output_en.txt
#SBATCH --partition=gpu_4
#SBATCH --ntasks=1
#SBATCH --time=18:00:00
#SBATCH --mem=40gb
#SBATCH --array=1-4  # Change the range to the number of instances you want
#SBATCH --gres=gpu:1
#SBATCH --job-name=dataaug-en

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
pip install pymongo transformers torch google-cloud-storage

# english Llama-2-GPTQ
pip install optimum>=1.12.0
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/  # Use cu117 if on CUDA 11.7

# # german Llama-2-GPTQ
# pip install packaging ninja
# pip install flash-attn==v1.0.9 --no-build-isolation

#pip install -r requirements_dataaug.txt

echo "Installed Dependencies"

ARRAY_INDEX=$SLURM_ARRAY_TASK_ID

python3 data_augmentation_english.py $ARRAY_INDEX >  py_output_en_$ARRAY_INDE.txt 2>&1

echo "Script ran through"
# Deactivate the virtual environment
deactivate