#!/bin/bash

models=(GCN GAT GIN GraphSage)
defense_modes=(none prune)
datasets=(Cora Pubmed Flickr ogbn-arxiv)

mkdir -p logs_clean

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for defense_mode in "${defense_modes[@]}"; do
            log_file="logs_clean/CA_${model}_${defense_mode}_${dataset}.log"

            echo "Running: model=$model, defense_mode=$defense_mode, dataset=$dataset"

            python -u ./run_clean.py  \
                --model "$model" \
                --dataset "$dataset" \
                --defense_mode "$defense_mode" \
                --device_id 0 \
            > "$log_file" 2>&1

            echo "Output saved to $log_file"
        done
    done
done
