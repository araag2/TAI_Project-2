#!/bin/bash

# Fixed parameters
PROMPT=prompts/OpenBookQA/OpenBookQA_2-shot_llama3.yaml
TASK=yaml_prompt_to_jsonl

# Data Files split by ":" where the first part is the dataset file and the second part is the output file
DATASETS=(
  "main-test.jsonl:main-train.jsonl:2-shot_main-test.jsonl"
  "main-dev.jsonl:main-train.jsonl:2-shot_main-dev.jsonl"
  "main-dev_train-LM.jsonl:main-train.jsonl:2-shot_main-dev_train-LM.jsonl"
  "main-train.jsonl:main-train.jsonl:2-shot_main-train.jsonl"
  "add-test.jsonl:add-train.jsonl:2-shot_add-test.jsonl"
  "add-dev.jsonl:add-train.jsonl:2-shot_add-dev.jsonl"
  "add-dev_train-LM.jsonl:add-train.jsonl:2-shot_add-dev_train-LM.jsonl"
  "add-train.jsonl:add-train.jsonl:2-shot_add-train.jsonl"
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