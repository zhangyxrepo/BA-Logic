#!/bin/bash

# Define arrays of datasets, models, and defense models
models=(GCN GAT GIN GraphSage)
defense_modes=(none prune)
datasets=(Cora Pubmed Flickr ogbn-arxiv)

# Create the logs_baseline directory if it doesn't exist
mkdir -p logs_baseline/GTA

# Loop through each combination of dataset, model, and defense model
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for defense_mode in "${defense_modes[@]}"; do
            log_file="logs_baseline/GTA/GTA_${dataset}_${model}_${defense_mode}.log"
            echo "Running run_GTA-C.py with dataset: $dataset, model: $model, defense model: $defense_mode"

            python -u ./run_GTA-C.py  \
                --test_model "$model" \
                --dataset "$dataset" \
                --defense_mode "$defense_mode" \
                --epochs 400 \
                --trojan_epochs 400 \
                --vs_number 500 \
                --device_id 1 \
            > "$log_file" 2>&1
        done
    done
done