#!/bin/bash

models=(GCN GAT GIN GraphSage)
defense_modes=(none prune)
datasets=(Cora)
vs_number=(250 300 350)
epochs=(250 300 400)
trojan_epochs=(250 300 400)

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
                            --device_id 0 \
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
