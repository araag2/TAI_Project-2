#!/bin/bash


# Using glob pattern to match all relevant files in the outputs directory. Can simplify the list or add more patterns as needed.
RES_FILES=(
  outputs/no_training/OpenBookQA/*/*/*.jsonl
  outputs/training/OpenBookQA/llama3/*/*/*.jsonl
)

echo -e "-------------------------------"
echo -e "Running eval for OpenBookQA dataset"
echo -e "-------------------------------\n"

for file in "${RES_FILES[@]}"; do
    IFS=":" read -r PREDICTION_FILE <<< $file

    gold_set_name=$(basename "$PREDICTION_FILE" .jsonl | awk -F'_' '{print $NF}')

    # Using the gold_set_name to construct the gold file path. Change as needed.Assuming the gold files end in "...add_dev.jsonl", "...add_test.jsonl", etc.
    GOLD_FILE="data/OpenBookQA/processed/$gold_set_name.jsonl"
    OUTPUT_DIR="$(dirname "$PREDICTION_FILE")/"

    python -m src.eval.eval_OpenBookQA \
        --prediction_file $PREDICTION_FILE \
        --gold_file $GOLD_FILE \
        --output_dir $OUTPUT_DIR

    PRINT_NAME=$(basename "$PREDICTION_FILE" .jsonl)

    echo "Done with $PREDICTION_FILE into SCORES-$PRINT_NAME.md"
    echo -e "-------------------------------\n"
done

echo "All files processed."