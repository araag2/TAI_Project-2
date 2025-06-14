#!/bin/bash

# Fixed parameters
PROMPT=prompts/OpenBookQA/OpenBookQA_Explain-Reasoning.yaml
TASK=yaml_prompt_to_jsonl

# Data Files split by ":" where the first part is the dataset file and the second part is the output file
DATASETS=(
  "main-train.jsonl:Train-Reasoning_main-train.jsonl"
  "add-train.jsonl:Train-Reasoning_add-train.jsonl"
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