#!/bin/bash

models=(GAT)
defense_modes=(none prune isolate)
datasets=("Cora" "Pubmed" "Flickr" "ogbn-arxiv")
vs_number=(150 200 300 400)
selection_methods=(none cluster_degree)


mkdir -p logs

for model in "${models[@]}"; do
for defense_mode in "${defense_modes[@]}"; do
for dataset in "${datasets[@]}"; do
        if [ "$dataset" == "Cora" ] || [ "$dataset" == "Flickr" ]; then
            num_classes=7
        elif [ "$dataset" == "Pubmed" ]; then
            num_classes=3
        elif [ "$dataset" == "ogbn-arxiv" ]; then
            num_classes=7
        fi
for vs_number in "${vs_number[@]}"; do
for selection_method in "${selection_methods[@]}"; do
for target_class in $(seq 0 $((num_classes - 1))); do
for poison_class in $(seq 0 $((num_classes - 1))); do
            if [ "$target_class" -ne "$poison_class" ]; then
                

 log_file="logs/${model}_${defense_mode}_${dataset}_wd${wd}_target${target_class}_poison${poison_class}.log"

 echo "Running: model=$model, defense_mode=$defense_mode, dataset=$dataset, vs_number=$vs_number, target_class=$target_class, poison_class=$poison_class"

 python -u ./run_clip.py  \
    --test_model "$model" \
    --defense_mode "$defense_mode" \
    --device_id 1 \
    --dataset "$dataset" \
    --vs_number "$vs_number" \
    --selection_method "$selection_method" \
    --epochs=700 \
    --trojan_epochs=400 \
    --target_class "$target_class" \
    --poison_class "$poison_class" \
 > "$log_file" 2>&1

 echo "Output saved to $log_file"
 fi
 done
 done
 done
 done
 done
 done
 done
