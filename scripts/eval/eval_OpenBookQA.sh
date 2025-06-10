#!/bin/bash

PRED_GOLD_OUTPUT_FILES=(
    
  "outputs/training/OpenBookQA/llama3/0-shot/training_0-shot_OpenBookQA_llama3_add-dev.jsonl:data/OpenBookQA/processed/add_dev.jsonl:outputs/training/OpenBookQA/llama3/0-shot/"
  "outputs/training/OpenBookQA/llama3/0-shot/training_0-shot_OpenBookQA_llama3_add-test.jsonl:data/OpenBookQA/processed/add_test.jsonl:outputs/training/OpenBookQA/llama3/0-shot/"
  "outputs/training/OpenBookQA/llama3/0-shot/training_0-shot_OpenBookQA_llama3_add-train.jsonl:data/OpenBookQA/processed/add_train.jsonl:outputs/training/OpenBookQA/llama3/0-shot/"
  "outputs/training/OpenBookQA/llama3/0-shot/training_0-shot_OpenBookQA_llama3_main-dev.jsonl:data/OpenBookQA/processed/main_dev.jsonl:outputs/training/OpenBookQA/llama3/0-shot/"
  "outputs/training/OpenBookQA/llama3/0-shot/training_0-shot_OpenBookQA_llama3_main-test.jsonl:data/OpenBookQA/processed/main_test.jsonl:outputs/training/OpenBookQA/llama3/0-shot/"
  "outputs/training/OpenBookQA/llama3/0-shot/training_0-shot_OpenBookQA_llama3_main-train.jsonl:data/OpenBookQA/processed/main_train.jsonl:outputs/training/OpenBookQA/llama3/0-shot/"
  # Add more triplets as needed
)

echo -e "-------------------------------"
echo -e "Running eval for OpenBookQA dataset"
echo -e "-------------------------------\n"

for triplet in "${PRED_GOLD_OUTPUT_FILES[@]}"; do
    IFS=":" read -r PREDICTION_FILE GOLD_FILE OUTPUT_DIR <<< $triplet

    #--checkpoint "" \
    #--no_sample \
    python -m src.eval.eval_OpenBookQA \
        --prediction_file $PREDICTION_FILE \
        --gold_file $GOLD_FILE \
        --output_dir $OUTPUT_DIR

    echo "Done with $PREDICTION_FILE into SCORES-$PREDICTION_FILE.jsonl"
    echo -e "-------------------------------\n"
done

echo "All files processed."