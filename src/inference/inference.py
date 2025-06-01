import argparse
import json
import torch
import random
import os
from datetime import datetime

# Model Libs
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer, set_seed
from datasets import load_dataset

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

def inference_OpenBookQA(args : argparse.Namespace):
    """
    Perform inference on the OpenBookQA dataset
    """
    dataset = load_dataset('json', data_files=f'{args.data}', split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    dataset = dataset.map(lambda x: {"messages": tokenizer.apply_chat_template(x["messages"], 
                                     tokenize=False, add_generation_prompt=True)})

    sampling_config=SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        #max_tokens=args.max_new_tokens,
        #num_return_sequences=args.num_return_sequences,
        guided_decoding= GuidedDecodingParams(
            choice=['A', 'B', 'C', 'D'])
    )

    model = LLM(model=args.model)

    outputs = model.generate(dataset["messages"], 
                            sampling_params=sampling_config, 
                            batch_size=args.batch_size
    )

    output_res = [{'id' : output.id, 'Prompt': output.prompt, 'Answer': output.outputs[0].text} for output in outputs]

    with safe_open_w(f"{args.output_dir}{args.exp_name}.jsonl") as output_file:
        for output in output_res:
            output_file.write(json.dumps(output, ensure_ascii=False, indent=4) + '\n')

def main():
    parser = argparse.ArgumentParser()

    # Model and checkpoint paths
    parser.add_argument('--model', type=str, help='name of the model used to generate and combine prompts', default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--exp_name', type=str, help='name of the experiment', default='OpenBookQA/0-shot/llama3/test_0-shot_conversation')

    #Very important, or it won't work with our formatting
    parser.add_argument('--dataset', type=str, help='dataset / experiment ran', default='OpenBookQA',
    choices=["OpenBookQA"])

    # Merge Params
    parser.add_argument('--checkpoint', type=str, help='path to model checkpoint, used if merging', default="")

    # Path to queries, qrels and prompt files
    #parser.add_argument('--used_set', type=str, help='choose which data to use', default='') # train | dev | test
    args = parser.parse_known_args()

    parser.add_argument('--data', type=str, help='path to file with data used to perform inference', default='../../data/OpenBookQA/inference/0-shot/0-shot_main_test.jsonl')

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

    # Output directory
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="outputs/")

    # Random Seed
    parser.add_argument('--random_seed', type=int, default=0)

    args = parser.parse_args()

    # Control Randomness for Reproducibility experiments
    set_random_seed(args.random_seed)

    model = None

    #if args.checkpoint != "":
    #    model = AutoModelForCausalLM.from_pretrained(args.model, device_map= {"": 0}, torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2")
    #    model = PeftModel.from_pretrained(model, args.checkpoint, device_map= {"": 0}, torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2")
    #    model = model.merge_and_unload()
    #    print(f'Merged {args.model=} with {args.checkpoint=}')
    #
    #else:
    #    model = AutoModelForCausalLM.from_pretrained(
    #        args.model, low_cpu_mem_usage=True,
    #        return_dict=True, torch_dtype=torch.bfloat16,
    #        device_map= {"": 0}, attn_implementation="flash_attention_2"
    #    )

    match args.dataset:
        case 'OpenBookQA':
            inference_OpenBookQA(args)
        case _:
            raise ValueError(f"Unknown dataset: {args.dataset}")

if __name__ == '__main__':
    main()