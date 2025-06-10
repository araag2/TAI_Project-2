MODEL=meta-llama/Llama-3.2-3B-Instruct
#CHECKPOINT= empty, bc 0 shot inference
DATASET=OpenBookQA 

# Generation Params
BATCH_SIZE=8 # Batch size for inference, doesn't really matter for 2-shot inference
MAX_NEW_TOKENS=10 # Max_New_Tokens to generate, doesn't really matter for MCQA without justification
TEMPERATURE=1
TOP_K=50
TOP_P=0.95
NUM_RETURN_SEQUENCES=1
SEED=0

#Ouput Dir
OUTPUT_DIR=outputs/no_training/OpenBookQA/llama3/few-shot/

# Data Files split by ":" where the first part is the experience name, and the second part is the path to the data file
DATA_FILES=(
  #"no-training_2-shot_OpenBookQA_llama3_main-train:data/OpenBookQA/inference/2-shot/2-shot_main-train.jsonl"
  "no-training_2-shot_OpenBookQA_llama3_main-dev:data/OpenBookQA/inference/few-shot/2-shot_main-dev.jsonl"
  "no-training_2-shot_OpenBookQA_llama3_main-test:data/OpenBookQA/inference/few-shot/2-shot_main-test.jsonl"
  #"no-training_2-shot_OpenBookQA_llama3_add-train:data/OpenBookQA/inference/2-shot/2-shot_add-train.jsonl"
  "no-training_2-shot_OpenBookQA_llama3_add-dev:data/OpenBookQA/inference/few-shot/2-shot_add-dev.jsonl"
  "no-training_2-shot_OpenBookQA_llama3_add-test:data/OpenBookQA/inference/few-shot/2-shot_add-test.jsonl"
)

echo -e "-------------------------------\n"
echo -e "Running 2-shot Inference with file src.inference.inference.py for:\n Dataset = $DATASET\n Model = $MODEL\n Output Dir = $OUTPUT_DIR\n"

for pair in "${DATA_FILES[@]}"; do
    IFS=":" read -r EXP_NAME DATA <<< $pair

    echo "Running $EXP_NAME, with data $DATA > outputs in < $OUTPUT_DIR $EXPNAME.jsonl"

    #--checkpoint "" \
    #--no_sample \
    CUDA_VISIBLE_DEVICES=$1 python -m src.inference.inference \
        --model $MODEL\
        --exp_name $EXP_NAME \
        --dataset $DATASET \
        --data $DATA \
        --batch_size $BATCH_SIZE \
        --max_new_tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE \
        --top_k $TOP_K \
        --top_p $TOP_P \
        --num_return_sequences $NUM_RETURN_SEQUENCES \
        --random_seed $SEED \
        --output_dir $OUTPUT_DIR

    echo "Done with $OUTPUT_DIR $EXPNAME.jsonl"
    echo -e "-------------------------------\n"
done

echo "All files processed."