#!/bin/bash

language="de"
# language="en"
# indexName="IndexGermanAllFiltered"
# indexName="IndexEnglishAllFiltered"
# indexName="translated_IndexEnglishAllFiltered"
# indexName="translated_IndexGermanAllFiltered"

# indexName="IndexEnglishAll_short"

# indexName="IndexEnglishAllFiltered_long"
# indexName="IndexGermanAllFiltered_long"
indexName="translated_IndexEnglishAllFiltered_long"
# indexName="translated_IndexGermanAllFiltered_long"

# dataset="qac"
dataset="qac_extended"
# dataset="generate_qac"


settings=(
  # "Llama2 1,2,3,5,10 True,False 1 5"
  # "LeoLama2 1,2,3,5,10 True,False 1 5"
  # "Llama2 1 False 1 20"
  # "LeoLama2 1 False 1 20"
#   "LeoLama2 1,2,3,5,10 True,False"
  "GPT3 1,2,3,5,10 True,False 1 1"
)

function join_by { local IFS="$1"; shift; echo "$*"; }

for setting in "${settings[@]}"; do
  # Split the setting into an array
  IFS=" " read -ra params <<< "${setting[@]}"

  reader="${params[0]}"

  # Extracting k_values as a list of integers
  IFS=',' read -ra cl <<< "${params[1]}"
  IFS=',' read -ra random <<< "${params[2]}"

  IFS=',' read -ra batchSize <<< "${params[3]}"

  split="${params[4]}"

  # Iterate over context_length
  for cl_value in "${cl[@]}"; do
    # Iterate over random_shuffle
    for random_value in "${random[@]}"; do

      # Iterate over batch_size
      for bs_value in "${batchSize[@]}"; do
        output_file="py_output/${reader}/${language}/${indexName}/py_output_${reader}_${language}_${indexName}_cl${cl_value}_random${random_value}_bs${bs_value}.txt"
        std_output_file="job_output/${reader}/${language}/${indexName}/output_${reader}_${language}_${indexName}_cl${cl_value}_random${random_value}_bs${bs_value}.txt"
        job_name="${reader}_dataset${dataset}_cl${cl_value}_${language}_${indexName}_random${random_value}_bs${bs_value}"

        # Skip if cl_value is 1 and random_value is True

        if [ "$cl_value" -eq 1 ] && [ "$random_value" = "True" ]; then
          continue
        fi

        echo "Executing next command for reader: ${reader} | context_length: ${cl_value} | random_shuffle: ${random_value} | indexName: ${indexName} | language: ${language} | dataset: ${dataset} | batch_size: ${bs_value}"

        # Check and create subfolders if they don't exist
        if [ ! -d "py_output/${reader}/${language}/${indexName}" ]; then
          mkdir -p "py_output/${reader}/${language}/${indexName}"
        fi

        if [ ! -d "job_output/${reader}/${language}/${indexName}" ]; then
          mkdir -p "job_output/${reader}/${language}/${indexName}"
        fi

        for splitIndex in $(seq 0 $((split-1))); do
          echo "SplitIndex: ${splitIndex} | split: ${split}"
          sbatch_options="--output=$std_output_file --job-name=$job_name --export=reader=$reader,cl=$cl_value,random=$random_value,language=$language,dataset=$dataset,indexName=$indexName,output_file=$output_file,batchSize=$bs_value,split=$split,splitIndex=$splitIndex --nodes=1"

          if [ "$reader" = "GPT3" ]; then
            sbatch $sbatch_options single_job.sh
          else
            sbatch $sbatch_options gpu_job.sh
          fi
        done
      done
    done
  done
done
