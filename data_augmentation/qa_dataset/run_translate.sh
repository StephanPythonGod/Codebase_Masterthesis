#!/bin/bash

#SBATCH --output=output_en.txt
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=40gb
#SBATCH --job-name=dataaug-en

# Load the modules
echo "Starting ..."

module load devel/python/3.8.6_gnu_10.2

export GOOGLE_APPLICATION_CREDENTIALS=/home/hd/hd_hd/hd_ff305/code/data_augmentation/qa_dataset/key.json

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

echo $GOOGLE_APPLICATION_CREDENTIALS

# dependencies
pip install google-cloud-translate==2.0.1

echo "Installed Dependencies"


python3 translate.py >  py_output.txt 2>&1

echo "Script ran through"
# Deactivate the virtual environment
deactivate