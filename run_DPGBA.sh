#!/bin/bash

models=(GCN GAT GIN GraphSage)
defense_modes=(none reconstruct)
datasets=(Cora Pubmed Flickr ogbn-arxiv)

# Create the logs_baseline directory if it doesn't exist
mkdir -p logs_baseline/DPGBA

# Loop through each combination of dataset, model, and defense model
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for defense_mode in "${defense_modes[@]}"; do
            log_file="logs_baseline/DPGBA/DPGBA_${dataset}_${model}_${defense_mode}.log"
            echo "Running run_DPGBA.py with dataset: $dataset, model: $model, defense model: $defense_mode"

            python -u /home/zhangyuxiang/clipgba/DPGBA-master/DPGBA/run/run_adaptive.py  \
                --test_model "$model" \
                --dataset "$dataset" \
                --defense_mode "$defense_mode" \
                --epochs 500 \
                --trojan_epochs 500 \
                --vs_number 500 \
                --device_id 3 \
            > "$log_file" 2>&1
        done
    done
done