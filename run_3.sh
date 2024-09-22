#!/bin/bash

models=(GCN GAT GIN GraphSage)
defense_modes=(none prune)
datasets=(ogbn-arxiv)
vs_number=(300 350 400)
epochs=(700 800)
trojan_epochs=(700 800)

mkdir -p logs

for model in "${models[@]}"; do
    for defense_mode in "${defense_modes[@]}"; do
        for dataset in "${datasets[@]}"; do
            for vs_number in "${vs_number[@]}"; do
                for epochs in "${epochs[@]}"; do
                    for trojan_epochs in "${trojan_epochs[@]}"; do
                        log_file="logs/${model}_${defense_mode}_${dataset}_vs${vs_number}_epochs${epochs}_trojan_epochs${trojan_epochs}.log"

                        echo "Running: model=$model, defense_mode=$defense_mode, dataset=$dataset, vs_number=$vs_number, epochs=$epochs, trojan_epochs=$trojan_epochs"

                        python -u ./run_clip.py  \
                            --test_model "$model" \
                            --defense_mode "$defense_mode" \
                            --device_id 3 \
                            --prune_thr 0.9\
                            --hidden 128 \
                            --dataset "$dataset" \
                            --vs_number "$vs_number" \
                            --epochs "$epochs" \
                            --trojan_epochs "$trojan_epochs" \
                        > "$log_file" 2>&1

                        echo "Output saved to $log_file"
                    done
                done
            done
        done
    done
 done
