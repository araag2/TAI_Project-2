#!/bin/bash

# Fixed parameters
PROMPT=prompts/OpenBookQA/OpenBookQA_CoT-0-shot_llama3.yaml
TASK=yaml_prompt_to_jsonl

# Data Files split by ":" where the first part is the dataset file and the second part is the output file
DATASETS=(
  "main-test.jsonl:CoT-0-shot_main-test.jsonl"
  "main-dev.jsonl:CoT-0-shot_main-dev.jsonl"
  "add-test.jsonl:CoT-0-shot_add-test.jsonl"
  "add-dev.jsonl:CoT-0-shot_add-dev.jsonl"
  "main-train.jsonl:CoT-0-shot_main-train.jsonl"
  "add-train.jsonl:CoT-0-shot_add-train.jsonl"
)

for pair in "${DATASETS[@]}"; do
    IFS=":" read -r dataset_file output_file <<< $pair

    # Construct full paths
    DATASET_FILE="data/OpenBookQA/processed/$dataset_file"
    OUTPUT_FILE="data/OpenBookQA/inference/CoT-0-shot/$output_file"

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