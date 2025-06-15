#!/bin/bash

# Fixed parameters
PROMPT=prompts/OpenBookQA/OpenBookQA_Multiple-Reasoning-Chains.yaml
TASK=yaml_prompt_to_jsonl

# Data Files split by ":" where the first part is the dataset file and the second part is the output file
DATASETS=(
  "main-test.jsonl:Multiple-Reasoning-Chains_main-test.jsonl"
  "main-dev.jsonl:Multiple-Reasoning-Chains_main-dev.jsonl"
  "add-test.jsonl:Multiple-Reasoning-Chains_add-test.jsonl"
  "add-dev.jsonl:Multiple-Reasoning-Chains_add-dev.jsonl"
  "main-train.jsonl:Multiple-Reasoning-Chains_main-train.jsonl"
  "add-train.jsonl:Multiple-Reasoning-Chains_add-train.jsonl"
)

for pair in "${DATASETS[@]}"; do
    IFS=":" read -r dataset_file output_file <<< $pair

    # Construct full paths
    DATASET_FILE="data/OpenBookQA/processed/$dataset_file"
    OUTPUT_FILE="data/OpenBookQA/inference/Reasoning/$output_file"

    echo "Processing $dataset_file > parsed into -> $output_file"

        python -m data.OpenBookQA.dataset-parser_OpenBookQA \
            --prompt $PROMPT \
            --dataset_file $DATASET_FILE \
            --output_file $OUTPUT_FILE \
            --task $TASK

    echo "Done with $output_file"
    echo -e "-------------------------------\n"
done

echo "All files processed."