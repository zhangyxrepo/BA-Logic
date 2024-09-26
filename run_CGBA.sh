#!/bin/bash

# Define arrays of datasets, models, and defense models
models=(GCN GAT GIN GraphSage)
defense_modes=(none prune)
datasets=(Cora Pubmed Flickr ogbn-arxiv)

# Create the logs_baseline directory if it doesn't exist
mkdir -p logs_baseline/CGBA

# Loop through each combination of dataset, model, and defense model
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for defense_mode in "${defense_modes[@]}"; do
            log_file="logs_baseline/CGBA/CGBA_${dataset}_${model}_${defense_mode}.log"
            echo "Running run_CGBA.py with dataset: $dataset, model: $model, defense model: $defense_mode"

            python -u ./run_CGBA.py  \
                --test_model "$model" \
                --dataset "$dataset" \
                --defense_mode "$defense_mode" \
                --vs_size 1200 \
                --device_id 0 \
            > "$log_file" 2>&1
        done
    done
done