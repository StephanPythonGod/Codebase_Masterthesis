#!/bin/bash

language="de"
# language="en"
# indexName="IndexGermanAll"
# indexName="IndexEnglishAll"
indexName="translated_IndexEnglishAll"
# indexName="translated_IndexGermanAll"

settings=(
  # "BM25 1,3,5,10,15,25,50,100,150,200 None False single"
  # "BM25 1,3,5,10,15,25,50,100,150,200,300,400,500,1000 None True single"
  # "BM25 1,3 None True single"
  # "BM25 300,400,500,1000 None False single"
  # "BM25 2000 None False order"
  # "BM25+CE 1,3,5,10,15,25,50,100,150,200,300,400,500,1000 1,3,5,10 True gpu"
  # "BM25+CE 300,400,500,1000 1,3,5,10 True gpu"
  # "BM25+CE 500,1000 1,3,5,10 True gpu"
  # "BM25+CE 1 1 True gpu"
  "DPR 200 None False order"
  # "DPR 500,1000 None False order"
)

function join_by { local IFS="$1"; shift; echo "$*"; }

for setting in "${settings[@]}"; do
  # Split the setting into an array
  IFS=" " read -ra params <<< "${setting[@]}"

  retriever="${params[0]}"

  # Extracting k_values as a list of integers
  IFS=',' read -ra k_values <<< "${params[1]}"
  IFS=',' read -ra k_outer_values <<< "${params[2]}" 

  ensemble="${params[3]}"
  partition="${params[4]}"

  # If k_outer_values is "None," set it to an empty array
  [[ $k_outer_values == "None" ]] && k_outer_values=()

  if [ -z "${k_outer_values[@]}" ]; then
    k_outer_values=(0)
  fi


  # Define common sbatch options

  if [ "$partition" == "order" ]; then
    k_values=${params[1]}
    modified_string=$(echo "$k_values" | tr ',' '-')
    output_file="py_output/${retriever}/${language}/${indexName}/py_output_${retriever}_${modified_string}_${ensemble}_${language}_${indexName}.txt"
    std_output_file="job_output/${retriever}/${language}/${indexName}/output_${retriever}_${modified_string}_${ensemble}_${language}_${indexName}.txt"
    echo "Executing next command for k: ${k_values} | k_outer: $k_outer_values"

    # Check and create subfolders if they don't exist
    if [ ! -d "py_output/${retriever}/${language}/${indexName}" ]; then
        mkdir -p "py_output/${retriever}/${language}/${indexName}"
    fi

    if [ ! -d "job_output/${retriever}/${language}/${indexName}" ]; then
        mkdir -p "job_output/${retriever}/${language}/${indexName}"
    fi
    sbatch_options="--output=$std_output_file --export=retriever=$retriever,k='${k_values}',k_outer=$k_outer_values,ensemble=$ensemble,language=$language,indexName=$indexName,output_file=$output_file --nodes=1"
    # echo $sbatch_options
    sbatch $sbatch_options single_job.sh
  else
    # Iterate over k_values
    for k in "${k_values[@]}"; do

      # Iterate over k_outer_values
      for k_outer in "${k_outer_values[@]}"; do
        # Skip iteration if k_outer is greater than k
        if [ "$k_outer" -gt "$k" ]; then
          continue
        fi

        output_file="py_output/${retriever}/${language}/${indexName}/py_output_${retriever}_${k}_${k_outer}_${ensemble}_${language}_${indexName}.txt"
        std_output_file="job_output/${retriever}/${language}/${indexName}/output_${retriever}_${k}_${k_outer}_${ensemble}_${language}_${indexName}.txt"
        echo "Executing next command for k: $k | k_outer: $k_outer"

        # Check and create subfolders if they don't exist
        if [ ! -d "py_output/${retriever}/${language}/${indexName}" ]; then
            mkdir -p "py_output/${retriever}/${language}/${indexName}"
        fi

        if [ ! -d "job_output/${retriever}/${language}/${indexName}" ]; then
            mkdir -p "job_output/${retriever}/${language}/${indexName}"
        fi


        sbatch_options="--output=$std_output_file --export=retriever=$retriever,k=$k,k_outer=$k_outer,ensemble=$ensemble,language=$language,indexName=$indexName,output_file=$output_file --nodes=1"

        if [ "$partition" == "single" ]; then
          sbatch $sbatch_options single_job.sh
        elif [ "$partition" == "gpu" ]; then
          sbatch $sbatch_options gpu_job.sh
        else
          echo "Unknown partition type: $partition"
        fi
      done
    done
  fi
done
