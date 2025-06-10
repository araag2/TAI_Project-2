#!/bin/bash
MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
TOKENIZER_NAME=meta-llama/Llama-3.2-3B-Instruct
EXP_NAME=Baseline-Training_0-shot_OpenBookQA_llama3_main
RUN=1
MERGE=false
CHECKPOINT=None
SAVE_DIR=models/Baseline-Training/OpenBookQA/llama3/
TRAIN_DATA=data/OpenBookQA/inference/0-shot/0-shot_main_train.jsonl
EVAL_DATA=data/OpenBookQA/inference/0-shot/0-shot_main_dev_train-LM.jsonl
TRAIN_TYPE=CompletionLM
LM_TOKEN=**Answer:**  # Token to indicate the start of the answer in the training data
# Hyperparameters
MAX_LENGTH=4000
BATCH_SIZE=1
POOLING=mean
TRAIN_EPOCHS=5
LR="2e-5"
WEIGHT_DECAY=0.00
# Lora Hyperparameters (if you turn on --lora)
LORA_R=128
LORA_DROPOUT=0.1
LORA_ALPHA=32
# Speed & memory optimizations
GRADIENT_ACCUMULATION_STEPS=4
# Options (yes/no) for: quant, lora, fp16, gradient_checkpointing
USE_QUANT=false
USE_LORA=true
USE_FP16=true
USE_GRADIENT_CHECKPOINTING=false

echo -e "-------------------------------"
echo -e "Running 0-shot CompletionLM training with file src.training.baseline_training for:"
echo -e "  Model      = $MODEL_NAME"
echo -e "  Train_Data = $TRAIN_DATA"
echo -e "  Eval_Data  = $EVAL_DATA"
echo -e "  Output Dir = $SAVE_DIR"
echo -e "-------------------------------"

CUDA_VISIBLE_DEVICES="$1" python -m src.training.training_script \
    --model_name $MODEL_NAME \
    --tokenizer_name $TOKENIZER_NAME \
    --exp_name $EXP_NAME \
    --run $RUN \
    \
    $( [ "$MERGE" == "true" ] && echo "--merge" ) \
    $( [ "$MERGE" == "false" ] && echo "--no-merge" ) \
    \
    --checkpoint $CHECKPOINT \
    \
    --save_dir $SAVE_DIR \
    \
    --train_data $TRAIN_DATA \
    --eval_data $EVAL_DATA \
    \
    --train_type $TRAIN_TYPE \
    --LM_Token $LM_TOKEN \
    \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --pooling $POOLING \
    --train_epochs $TRAIN_EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    \
    $( [ "$USE_QUANT" == "true" ] && echo "--quant" ) \
    $( [ "$USE_QUANT" == "false" ] && echo "--no-quant" ) \
    \
    $( [ "$USE_LORA" == "true" ] && echo "--lora" ) \
    $( [ "$USE_LORA" == "false" ] && echo "--no-lora" ) \
    --lora_r $LORA_R \
    --lora_dropout $LORA_DROPOUT \
    --lora_alpha $LORA_ALPHA \
    \
    $( [ "$USE_FP16" == "true" ] && echo "--fp16" ) \
    $( [ "$USE_FP16" == "false" ] && echo "--no-fp16" ) \
    \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    \
    $( [ "$USE_GRADIENT_CHECKPOINTING" == "true" ] && echo "--gradient_checkpointing" ) \
    $( [ "$USE_GRADIENT_CHECKPOINTING" == "false" ] && echo "--no-gradient_checkpointing" )

echo "Done with Training. Check output in $SAVE_DIR/end_model"
echo -e "-------------------------------"