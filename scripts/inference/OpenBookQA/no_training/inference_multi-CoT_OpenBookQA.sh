MODEL=google/gemma-3-4b-it
#CHECKPOINT= empty, bc 0 shot inference
DATASET=OpenBookQA 

# Generation Params
BATCH_SIZE=4 # Batch size for inference, doesn't really matter for 0-shot inference
MAX_NEW_TOKENS=1500 # Max_New_Tokens to generate, doesn't really matter for MCQA without justification
TEMPERATURE=1
TOP_K=50
TOP_P=0.95
NUM_RETURN_SEQUENCES=1
SEED=0

INFERENCE_TYPE=multi_reasoning

#Ouput Dir
OUTPUT_DIR=outputs/no_training/OpenBookQA/gemma-4B/multi-CoT/

# Data Files split by ":" where the first part is the experience name, and the second part is the path to the data file
DATA_SPLITS=(
  "main-test:Multiple-Reasoning-Chains_main-test"
  "main-dev:Multiple-Reasoning-Chains_main-dev"
  "add-test:Multiple-Reasoning-Chains_add-test"
  "add-dev:Multiple-Reasoning-Chains_add-dev"
)

echo -e "-------------------------------\n"
echo -e "Running multi-CoT Inference with file src.inference.inference.py for:\n Dataset = $DATASET\n Model = $MODEL\n Output Dir = $OUTPUT_DIR\n"

for pair in "${DATA_SPLITS[@]}"; do
    IFS=":" read -r data_split data_split_name <<< $pair

    EXP_NAME="no-training_multi-CoT_OpenBookQA_gemma-4B_$data_split"
    DATA="data/OpenBookQA/inference/Reasoning/$data_split_name.jsonl"

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
        --inference_type $INFERENCE_TYPE \
        --random_seed $SEED \
        --output_dir $OUTPUT_DIR

    echo "Done with $OUTPUT_DIR $EXPNAME.jsonl"
    echo -e "-------------------------------\n"
done

echo "All files processed."