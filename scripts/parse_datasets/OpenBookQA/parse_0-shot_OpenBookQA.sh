#!/bin/bash

# Fixed parameters
PROMPT=prompts/OpenBookQA/OpenBookQA_0-shot_llama3.yaml
TASK=yaml_prompt_to_jsonl

# Data Files split by ":" where the first part is the dataset file and the second part is the output file
DATASETS=(
  "main_test.jsonl:0-shot_main_test.jsonl"
  "main_dev.jsonl:0-shot_main_dev.jsonl"
  "main_dev_train-LM.jsonl:0-shot_main_dev_train-LM.jsonl"
  "main_train.jsonl:0-shot_main_train.jsonl"
  "add_test.jsonl:0-shot_add_test.jsonl"
  "add_dev.jsonl:0-shot_add_dev.jsonl"
  "add_dev_train-LM.jsonl:0-shot_add_dev_train-LM.jsonl"
  "add_train.jsonl:0-shot_add_train.jsonl"
)

for pair in "${DATASETS[@]}"; do
    IFS=":" read -r dataset_file output_file <<< $pair

    # Construct full paths
    DATASET_FILE="data/OpenBookQA/processed/$dataset_file"
    OUTPUT_FILE="data/OpenBookQA/inference/0-shot/$output_file"

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