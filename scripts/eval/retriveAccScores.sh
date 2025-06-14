#!/bin/bash

echo -e "-------------------------------"
echo -e "Retrieving Acc Socres for OpenBookQA dataset"
echo -e "-------------------------------\n"


python -m src.eval.retrieveAccScores \
    --top_dir outputs/ \
    --output_dir outputs/scores_csv/

echo -e "-------------------------------\n"

echo "All files processed."