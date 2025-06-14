import multiprocessing as mp
import argparse
import json
import torch
import random
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from datetime import datetime

# Model Libs
from collections import Counter
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer, set_seed, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from peft import PeftModel

def safe_open_w(path: str) -> object:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def set_random_seed(random_seed = 0):
    random.seed(random_seed)
    set_seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

def check_merge(args: argparse.Namespace) -> None:
    """
    Check if the model is merged or not, by checking if in the checkpoint path there is a file named `pytorch_model.bin`
    If not, it will merge the model and save it in the checkpoint path.
    """
    if os.path.exists(os.path.join(args.checkpoint, "pytorch_model.bin")) or \
           os.path.exists(os.path.join(args.checkpoint, "model.safetensors")):
        print(f"✅ Merged model already exists at '{args.checkpoint}'. Skipping merge.")
    else:
        print(f"🔄 Merging LoRA from '{args.checkpoint}' into base model '{args.model}'...")

        base_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
        model = model.merge_and_unload()  # Merges LoRA into base model

        # Save merged model
        model.save_pretrained(args.checkpoint)
        print(f"✅ Merged model saved to '{args.checkpoint}'.")
    return 

def generate_with_sampling_params(args : argparse.Namespace, dataset : Dataset, decodingParams : GuidedDecodingParams):
    sampling_config=SamplingParams(
        temperature = args.temperature,
        top_k = args.top_k,
        top_p = args.top_p,
        max_tokens = args.max_new_tokens,
        seed = args.random_seed if args.num_return_sequences == 1 else None,
        n = args.num_return_sequences,
        guided_decoding = decodingParams
    )

    quant_args = {}
    if args.quantization_type is not "":
        quant_args = {
            "quantization": "bitsandbytes",
            "dtype": args.quantization_type
        }

    model = LLM(model=args.model if args.checkpoint == "" else args.checkpoint,
        tokenizer=args.model, 
        tokenizer_mode="mistral" if "mistral" in args.model or "mixtral" in args.model else "auto", 
        **quant_args
    )
    
    return model.generate(dataset["messages"], sampling_params=sampling_config)

def self_consistency_inference(outputs : list) -> list:
    final_outputs = []
    possible_answers = {'A', 'B', 'C', 'D'}

    for result in outputs:
        answers = []
        for output in result.outputs:
            if output.text[-1] in possible_answers:
                answers.append(output.text[-1])

        final_answer = random.choice(list(possible_answers)) if not answers else None

        if answers:
            vote_counts = Counter(answers)
            most_common = vote_counts.most_common()

            top_choices = [ans for ans, count in most_common if count == most_common[0][1]]
            final_answer = random.choice(top_choices) # Randomly select among the most common answers

        final_outputs.append({
            'Prompt': result.prompt,
            'Answer': final_answer,
            'Votes': dict(vote_counts)
        })

    return final_outputs

def no_reasoning_OpenBookQA(args : argparse.Namespace, dataset=None):
    outputs = generate_with_sampling_params(args, dataset, GuidedDecodingParams(choice = ['A', 'B', 'C', 'D']))

    if args.num_return_sequences > 1:
        print(f"----USER MESSAGE----\nSince num_return_sequences is set to {args.num_return_sequences}, self-consistency inference will be performed instead of single sequence inference.----END----\n")

        return self_consistency_inference(outputs)

    return [{'Prompt': output.prompt, 'Answer': output.outputs[0].text} for output in outputs]


def CoT_OpenBookQA(args : argparse.Namespace, dataset=None):
    outputs = generate_with_sampling_params(args, dataset, GuidedDecodingParams(regex = r"Let's think step by step: [\s\S]*\n\n\*\*Answer:\*\* [A-D]"))

    if args.num_return_sequences > 1:
        print(f"----USER MESSAGE----\nSince num_return_sequences is set to {args.num_return_sequences}, self-consistency inference will be performed instead of single sequence inference.----END----\n")

        return self_consistency_inference(outputs)

    return [{'Prompt': output.prompt, 'Answer': output.outputs[0].text} for output in outputs]


def inference_OpenBookQA(args : argparse.Namespace):
    """
    Perform inference on the OpenBookQA dataset
    """
    dataset = load_dataset('json', data_files=f'{args.data}', split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    dataset = dataset.map(lambda x: {"messages": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=True)})

    match args.inference_type:
        case 'no_reasoning':
            output_res = no_reasoning_OpenBookQA(args, dataset)
        case 'CoT_reasoning':
            output_res = CoT_OpenBookQA(args, dataset)
        case 'self-refinement':
            pass
        case _:
            raise ValueError(f"Unknown inference type: {args.inference_type}")

    with safe_open_w(f"{args.output_dir}{args.exp_name}.jsonl") as output_file:
        for output in output_res:
            output_file.write(json.dumps(output, ensure_ascii=False) + '\n')

def main():
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()

    # Model and checkpoint paths
    parser.add_argument('--model', type=str, help='name of the model used to generate and combine prompts', default='meta-llama/Llama-3.2-3B-Instruct')
    # Merge Params
    parser.add_argument('--checkpoint', type=str, help='path to model checkpoint, used if merging', default="")

    parser.add_argument('--exp_name', type=str, help='name of the experiment', default='')

    #Very important, or it won't work with our formatting
    parser.add_argument('--dataset', type=str, help='dataset / experiment ran', default='OpenBookQA',
    choices=["OpenBookQA"])
    parser.add_argument('--data', type=str, help='path to file with data used to perform inference', default='../../data/OpenBookQA/inference/0-shot/0-shot_main_test.jsonl')

    # Path to queries, qrels and prompt files
    #parser.add_argument('--used_set', type=str, help='choose which data to use', default='') # train | dev | test
    #args = parser.parse_known_args()

    # Generation Params
    parser.add_argument('--batch_size', type=int, help='batch_size of generated examples', default=8)

    parser.add_argument('--sample', dest='sample', action='store_true')
    parser.add_argument('--no_sample', dest='sample', action='store_false')
    parser.set_defaults(sample=True)

    parser.add_argument('--max_new_tokens', type=int, help='sets the number of new tokens to generate when decoding', default=10)
    parser.add_argument('--temperature', type=float, help='generation param that sets the model stability', default=1)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--num_return_sequences', type=int, default=1)

    parser.add_argument('--quantization_type', type=str, help='quantization type to use for the model', default='')

    parser.add_argument('--inference_type', type=str, help='type of inference to perform', default='no_reasoning', choices=['no_reasoning', 'CoT_reasoning', 'self-refinement'])  

    # Random Seed
    parser.add_argument('--random_seed', type=int, default=0)

    # Output directory
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="outputs/")
    args = parser.parse_args()

    # Control Randomness for Reproducibility experiments
    set_random_seed(args.random_seed)

    if args.checkpoint != "":
        check_merge(args)

    match args.dataset:
        case 'OpenBookQA':
            inference_OpenBookQA(args)
        case _:
            raise ValueError(f"Unknown dataset: {args.dataset}")

if __name__ == '__main__':
    main()