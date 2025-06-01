#!/bin/bash

# Fixed parameters
SOURCE_FOLDER="data/OpenBookQA/source/"
OUTPUT_FOLDER="data/OpenBookQA/processed/"

# Parse OpenBookQA Base Dataset
echo "Parsing OpenBookQA: source_to_jsonl"

python -m data.OpenBookQA.dataset-parser_OpenBookQA \
    --source_folder $SOURCE_FOLDER \
    --task source_to_jsonl

echo -e "\nsource_to_jsonl Completed!\n\nParsing OpenBookQA: source_to_processed_jsonl"

python -m data.OpenBookQA.dataset-parser_OpenBookQA \
    --source_folder $SOURCE_FOLDER \
    --output_folder $OUTPUT_FOLDER \
    --task source_to_processed_jsonl

echo -e "\nsource_to_processed_jsonl Completed!\n"
echo -e "-------------------------------\nAll files processed."