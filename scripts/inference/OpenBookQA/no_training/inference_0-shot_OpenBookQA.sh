MODEL=google/gemma-3-4b-it
#CHECKPOINT= empty, bc 0 shot inference
DATASET=OpenBookQA 

# Generation Params
BATCH_SIZE=8 # Batch size for inference, doesn't really matter for 0-shot inference
MAX_NEW_TOKENS=10 # Max_New_Tokens to generate, doesn't really matter for MCQA without justification
TEMPERATURE=1
TOP_K=50
TOP_P=0.95
NUM_RETURN_SEQUENCES=1
SEED=0

#Ouput Dir
OUTPUT_DIR=outputs/no_training/OpenBookQA/gemma-4B/0-shot/

# Data Files split by ":" where the first part is the experience name, and the second part is the path to the data file
DATA_SPLITS=(
  "main-dev:0-shot_main-dev"
  "main-test:0-shot_main-test"
  "add-dev:0-shot_add-dev"
  "add-test:0-shot_add-test"
)

echo -e "-------------------------------\n"
echo -e "Running 0-shot Inference with file src.inference.inference.py for:\n Dataset = $DATASET\n Model = $MODEL\n Output Dir = $OUTPUT_DIR\n"

for pair in "${DATA_SPLITS[@]}"; do
    IFS=":" read -r data_split data_split_name <<< $pair

    EXP_NAME="no-training_0-shot_OpenBookQA_gemma-4B_$data_split"
    DATA="data/OpenBookQA/inference/0-shot/$data_split_name.jsonl"

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