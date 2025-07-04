MODEL=Qwen/Qwen3-8B
#CHECKPOINT= empty, bc 0 shot inference
DATASET=OpenBookQA 

# Generation Params
BATCH_SIZE=1 # Batch size for inference, doesn't really matter for 0-shot inference
MAX_NEW_TOKENS=1500 # Max_New_Tokens to generate, doesn't really matter for MCQA without justification
TEMPERATURE=1
TOP_K=50
TOP_P=0.95
NUM_RETURN_SEQUENCES=1
SEED=0

#QUANTIZATION_TYPE=bfloat16
INFERENCE_TYPE=CoT_reasoning

#Ouput Dir
OUTPUT_DIR=outputs/no_training/OpenBookQA/Qwen-8B/Reasoning/

# Data Files split by ":" where the first part is the experience name, and the second part is the path to the data file
DATA_SPLITS=(
  "main-train:Train-Reasoning_main-train"
  "add-train:Train-Reasoning_add-train"
)

echo -e "-------------------------------\n"
echo -e "Running Reasoning 8-bit with file src.inference.inference.py for:\n Dataset = $DATASET\n Model = $MODEL\n Output Dir = $OUTPUT_DIR\n"

for pair in "${DATA_SPLITS[@]}"; do
    IFS=":" read -r data_split data_split_name <<< $pair

    EXP_NAME="Reasoning_8-bit_OpenBookQA_Qwen-14B_$data_split"
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
        #--quantization_type $QUANTIZATION_TYPE \

    echo "Done with $OUTPUT_DIR $EXPNAME.jsonl"
    echo -e "-------------------------------\n"
done

echo "All files processed."