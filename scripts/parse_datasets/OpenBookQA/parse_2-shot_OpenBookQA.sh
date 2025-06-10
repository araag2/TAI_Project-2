#!/bin/bash

# Fixed parameters
PROMPT=prompts/OpenBookQA/OpenBookQA_2-shot_llama3.yaml
TASK=yaml_prompt_to_jsonl

# Data Files split by ":" where the first part is the dataset file and the second part is the output file
DATASETS=(
  "main_test.jsonl:main_train.jsonl:2-shot_main_test.jsonl"
  "main_dev.jsonl:main_train.jsonl:2-shot_main_dev.jsonl"
  "main_dev_train-LM.jsonl:main_train.jsonl:2-shot_main_dev_train-LM.jsonl"
  "main_train.jsonl:main_train.jsonl:2-shot_main_train.jsonl"
  "add_test.jsonl:add_train.jsonl:2-shot_add_test.jsonl"
  "add_dev.jsonl:add_train.jsonl:2-shot_add_dev.jsonl"
  "add_dev_train-LM.jsonl:add_train.jsonl:2-shot_add_dev_train-LM.jsonl"
  "add_train.jsonl:add_train.jsonl:2-shot_add_train.jsonl"
)

for triplet in "${DATASETS[@]}"; do
    IFS=":" read -r dataset_file example_file output_file <<< $triplet 

    # Construct full paths
    DATASET_FILE="data/OpenBookQA/processed/$dataset_file"
    EXAMPLE_FILE="data/OpenBookQA/processed/$example_file"
    OUTPUT_FILE="data/OpenBookQA/inference/few-shot/$output_file"

    echo "Processing $dataset_file with examples $example_file > parsed into -> $output_file"

        python -m data.OpenBookQA.dataset-parser_OpenBookQA \
            --prompt $PROMPT \
            --dataset_file $DATASET_FILE \
            --example_file $EXAMPLE_FILE \
            --output_file $OUTPUT_FILE \
            --task $TASK

    echo "Done with $output_file"
    echo -e "-------------------------------\n"
done

echo "All files processed."