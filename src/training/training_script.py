import os
import wandb
import json
import torch
import argparse
import typing

# Local Files
from ..utils.file_utils import create_path

# Util libs
from datasets.arrow_dataset import Dataset

# Model Libs
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, DPOConfig, DPOTrainer, apply_chat_template

def completion_LM_training(args, train_dataset, eval_dataset, model, peft_config, tokenizer):
    training_arguments = TrainingArguments(
        #max_length = args.max_length,
        output_dir = args.save_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch" if eval_dataset else None,
        save_strategy="epoch",
        save_total_limit= 3,
        num_train_epochs = args.train_epochs,
        per_device_train_batch_size= args.batch_size,
        optim = "paged_adamw_8bit",
        logging_steps= 25,
        learning_rate= args.lr,
        weight_decay= args.weight_decay,
        bf16= False,
        group_by_length= True,
        lr_scheduler_type= "constant",
        #model load
        load_best_model_at_end= True if eval_dataset else False,
        #Speed and memory optimization parameters
        gradient_accumulation_steps= args.gradient_accumulation_steps,
        gradient_checkpointing= args.gradient_checkpointing,
        fp16= args.fp16,
        report_to="wandb"
    )
    
    print(train_dataset)

    ## Data collator for completing with "Answer: YES" or "Answer: NO"
    collator = DataCollatorForCompletionOnlyLM(args.LM_Token, tokenizer= tokenizer)

    ## Setting sft parameters
    trainer = SFTTrainer(
        model= model,
        data_collator= collator,
        train_dataset= train_dataset,
        eval_dataset= eval_dataset,
        peft_config= peft_config,
        #max_seq_length= args.max_length,
        #dataset_text_field= "text",
        #tokenizer= tokenizer,
        args= training_arguments,
        #packing= False,
    )

    return trainer

def regular_LM_training(args, train_dataset, eval_dataset, model, peft_config, tokenizer):
    training_arguments = TrainingArguments(
        output_dir = args.save_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=args.train_epochs,
        per_device_train_batch_size=args.batch_size,
        optim="paged_adamw_8bit",
        logging_steps=25,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        bf16=False,
        group_by_length=True,
        lr_scheduler_type="constant",
        load_best_model_at_end=True if eval_dataset else False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        report_to="wandb"
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Since it's a causal LM
    )

    trainer = SFTTrainer(
        model=model,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=training_arguments,
    )

    return trainer

def DPO_training(args, train_dataset, eval_dataset, model, peft_config, tokenizer):
    pass

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default="", help='model to train')
    parser.add_argument('--tokenizer_name', type=str, default="", help='tokenizer to use for the model')
    parser.add_argument('--exp_name', type=str, default="", help='Describes the conducted experiment')
    parser.add_argument('--run', type=int, default=1, help='run number for wandb logging')

    parser.add_argument('--merge', dest='merge', action='store_true', help='boolean flag to set if model is merging')
    parser.add_argument('--no-merge', dest='merge', action='store_false', help='boolean flag to set if model is merging')
    parser.set_defaults(merge=False)

    parser.add_argument('--checkpoint', type=str, help='path to model checkpoint, used if merging', default="")

    # I/O paths for models, CT, queries and qrels
    parser.add_argument('--save_dir', type=str, default="models/pre-train-complete-eligibility_plus_base-task-tamplate/", help='path to model save dir')

    parser.add_argument("--train_data", default="", type=str)
    parser.add_argument("--eval_data", default="", type=str)

    parser.add_argument("--train_type", default="CompletionLM", type=str, help="type of training", choices = ["CompletionLM", "RegularLM", "DPO"])

    # Data collator parameters
    parser.add_argument("--LM_Token", default="Answer:", type=str, help="Token to complete from")

    #Model Hyperparamenters
    parser.add_argument("--max_length", type=int, default=7000)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--train_epochs", default=5, type=int)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.00)

    # Quantization parameters
    parser.add_argument('--quant', dest='quant', action='store_true', help='boolean flag to set if LoRA is used')
    parser.add_argument('--no-quant', dest='quant', action='store_false', help='boolean flag to set if LoRA is used')
    parser.set_defaults(quant=True)


    # Lora Hyperparameters
    parser.add_argument('--lora', dest='lora', action='store_true', help='boolean flag to set if LoRA is used')
    parser.add_argument('--no-lora', dest='lora', action='store_false', help='boolean flag to set if LoRA is used')
    parser.set_defaults(lora=True)

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_alpha", type=float, default=16)

    #Speed and memory optimization parameters
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision instead of 32-bit")
    parser.add_argument("--no-fp16", dest='fp16', action="store_false", help="Whether to use 16-bit (mixed) precision instead of 32-bit")
    parser.set_defaults(fp16=True)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--gradient_checkpointing", action="store_true", help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--no-gradient_checkpointing", dest='gradient_checkpointing', action="store_false", help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.set_defaults(gradient_checkpointing=False)
    args = parser.parse_args()

    return args

def create_model_and_tokenizer(args : argparse):
    bnb_config = None
    if args.quant:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit= True,
            bnb_4bit_quant_type= "nf4",
            bnb_4bit_compute_dtype= torch.bfloat16,
            bnb_4bit_use_double_quant= False,
        )

    peft_config = None
    if args.lora:
        peft_config = LoraConfig(
            r = args.lora_r,
            lora_alpha= args.lora_alpha,
            lora_dropout= args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj"],
        )

    
    model = None
    if args.merge:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config= bnb_config, device_map= {"": 0}, torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2")
        
        model = PeftModel.from_pretrained(model, args.checkpoint, quantization_config= bnb_config, device_map= {"": 0}, torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2")

        model = model.merge_and_unload()

    else:
       model = AutoModelForCausalLM.from_pretrained(
            args.model_name, low_cpu_mem_usage=True,
            quantization_config= bnb_config,
            return_dict=True, torch_dtype=torch.bfloat16,
            attn_implementation='eager' if "gemma" in args.model_name else "flash_attention_2",
            device_map= {"": 0}
       )

    #### LLAMA STUFF
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    if peft_config is not None:
        model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    return model, peft_config, tokenizer

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()

    wandb.init(
        project="TAI-Report-2",
        name = f'{args.model_name}/{args.exp_name}/run-{args.run}',
        group = f'{args.model_name}/{args.exp_name}',
        config = { arg : getattr(args, arg) for arg in vars(args)}
    )

    # Load tokenizer and model
    model, peft_config, tokenizer = create_model_and_tokenizer(args)

    def format_dataset(data_file, tokenizer):
        raw_hf_ds = load_dataset('json', data_files=f'{data_file}', split="train")

        formatted_examples = [
            apply_chat_template(example, tokenizer=tokenizer)
            for example in raw_hf_ds
        ]

        if args.train_type == "CompletionLM" or args.train_type == "RegularLM":
            formatted_examples = [{"text" : f'{example["prompt"]}{example["completion"]}'} for example in formatted_examples]

        return Dataset.from_list(formatted_examples)

    train_dataset = format_dataset(args.train_data, tokenizer)
    eval_dataset = format_dataset(args.eval_data, tokenizer) if args.eval_data != "" else None
    #train_dataset = load_dataset('json', data_files=f'{args.train_data}', split="train")
    #eval_dataset = load_dataset('json', data_files=f'{args.eval_data}', split="train") if args.eval_data != "" else None

    trainer = None
    if args.train_type == "CompletionLM":
        trainer = completion_LM_training(args, train_dataset, eval_dataset, model, peft_config, tokenizer)

    elif args.train_type == "RegularLM":
        trainer = regular_LM_training(args, train_dataset, eval_dataset, model, peft_config, tokenizer)

    elif args.train_type == "DPO":
        trainer = DPO_training(args, train_dataset, eval_dataset, model, peft_config, tokenizer)

    ## Training
    trainer.train()

    ## Save model and finish run
    create_path(f'{args.save_dir}end_model/')
    trainer.model.save_pretrained(f'{args.save_dir}end_model/')
    wandb.finish()


if __name__ == '__main__':
    main()