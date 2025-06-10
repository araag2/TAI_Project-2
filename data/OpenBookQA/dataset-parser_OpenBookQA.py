import argparse
import typing
import yaml
import json
import copy
import os

from datasets import load_dataset

def safe_open_w(path: str) -> object:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def source_to_jsonl(source_path: str = "source/"):
    main_train = load_dataset("parquet", data_files=f"{source_path}main_train.parquet", split="train")
    add_train = load_dataset("parquet", data_files=f"{source_path}add_train.parquet", split="train")

    main_test = load_dataset("parquet", data_files=f"{source_path}main_test.parquet", split="train")
    add_test = load_dataset("parquet", data_files=f"{source_path}add_test.parquet", split="train")

    main_dev = load_dataset("parquet", data_files=f"{source_path}main_dev.parquet", split="train")
    add_dev = load_dataset("parquet", data_files=f"{source_path}add_dev.parquet", split="train")

    #output_to_jsonl
    main_train.to_json(source_path + "main-train.jsonl")
    add_train.to_json(source_path + "add-train.jsonl")

    main_test.to_json(source_path + "main-test.jsonl")
    add_test.to_json(source_path + "add-test.jsonl")

    main_dev.to_json(source_path + "main-dev.jsonl")
    add_dev.to_json(source_path + "add-dev.jsonl")

key_to_option = {0 : "A", 1 : "B", 2 : "C", 3 : "D"}

def source_to_processed_jsonl(source_path: str = "source/", output_path: str = "processed/"):
    main_train = load_dataset("parquet", data_files=f"{source_path}main_train.parquet", split="train")
    add_train = load_dataset("parquet", data_files=f"{source_path}add_train.parquet", split="train")

    main_test = load_dataset("parquet", data_files=f"{source_path}main_test.parquet", split="train")
    add_test = load_dataset("parquet", data_files=f"{source_path}add_test.parquet", split="train")

    main_dev = load_dataset("parquet", data_files=f"{source_path}main_dev.parquet", split="train")
    add_dev = load_dataset("parquet", data_files=f"{source_path}add_dev.parquet", split="train")

     # Columns names in the dataset [id, question_stem, choices [text, labels], answerKey] and ADD_ONLY [fact1, humanScore, clarity, turkIdAnonymized]

    for name, dataset in zip(["main-train", "add-train", "main-test", "add-test", "main-dev", "add-dev"],
                             [main_train, add_train, main_test, add_test, main_dev, add_dev]):
        
        dataset = dataset.rename_column("question_stem", "Question")
        dataset = dataset.rename_column("answerKey", "Label")

        # Go into column Choices, and divide text into 4 fields: Text_Answer_1, Text_Answer_2, Text_Answer_3, Text_Answer_4
        choices = dataset["choices"]
        for i in range(4):
            dataset = dataset.add_column(f"Option_{key_to_option[i]}", [choice["text"][i] for choice in choices])
        dataset = dataset.remove_columns(["choices"])

        if "add" in name:
            dataset = dataset.rename_column("fact1", "Support_Fact")
            dataset = dataset.rename_column("humanScore", "Human_Score")
            dataset = dataset.rename_column("clarity", "Clarity_Score")
            dataset = dataset.remove_columns(["turkIdAnonymized"])

        dataset.to_json(f"{output_path}{name}.jsonl", lines=True)

def replace_placeholders(iter_prompt: typing.List[dict], entry: dict):
    """
    Replace placeholders in the prompt with actual values from the entry.
    """
    for message in iter_prompt:
        if message['content']:
            for key, value in entry.items():
                message['content'] = message['content'].replace(f"{{{{{key}}}}}", str(value))
    return iter_prompt

def check_if_train_and_format(iter_prompt: typing.List[dict], entry : dict, file_name : str):
    """
    Check if the file is a training file and format the prompt accordingly.
    """
    if "train" not in file_name and iter_prompt[-1]['role'] == 'assistant':
        iter_prompt.pop(-1)
        return {"id": entry["id"], 
                "messages": iter_prompt, 
                "Label": entry["Label"]}
    
    elif "train" in file_name and iter_prompt[-1]['role'] == 'assistant':
        return {"id": entry["id"], 
                "prompt": iter_prompt[:-1], 
                "completion": [iter_prompt[-1]], 
                "Label": entry["Label"]}
    else:
        raise ValueError(f"Unexpected format in dataset file: {file_name}")

def zero_shot_conversation_prompt_to_jsonl(loaded_prompt : dict, dataset_file : typing.List[str], output_file : str):
    with load_dataset("json", data_files=dataset_file, split="train") as dataset:
        # Iterate over the dataset and create a new dictionary with the prompt format
        populated_prompts = []

        for entry in dataset:
            # Replace placeholders in the prompt with actual values from the entry
            iter_prompt = replace_placeholders(copy.deepcopy(loaded_prompt)["messages"] if "main" in dataset_file else copy.deepcopy(loaded_prompt)["messages_add"], entry)

            populated_prompts.append(check_if_train_and_format(iter_prompt, entry, dataset_file))            

    # Write the populated prompts to the output file
    with safe_open_w(output_file) as f_out:
        for item in populated_prompts:
            f_out.write(json.dumps(item) + "\n")

def process_examples(loaded_prompt: dict, example_dataset_file: str) -> typing.List[dict]:
    """
    Process the examples from the example dataset file and return a list of plain text prompts to use as few-shot examples.
    """
    example_prompts = []

    with load_dataset("json", data_files=example_dataset_file, split="train") as example_dataset:
        for entry in example_dataset:
            # Replace placeholders in the prompt with actual values from the entry
            iter_prompt = replace_placeholders(copy.deepcopy(loaded_prompt)["messages_single"] if "main" in example_dataset_file else copy.deepcopy(loaded_prompt)["messages_single_add"], entry)

            if iter_prompt[0]['role'] == 'system':
                iter_prompt.pop(0)

            if iter_prompt[0]['role'] == 'user':
                iter_prompt[0]['content'] = f"Example {{{{shot_number}}}}:\n\n{iter_prompt[0]['content']}"

            example_prompts.append(iter_prompt)

        # Create few-shot examples bundles with the specified number of examples, replacing the {{shot_number}} placeholder
        example_bundles = [copy.deepcopy(example_prompts[i:] + example_prompts[:i]) for i in range(loaded_prompt["shot_number"])]

        for i in range(len(example_bundles)):
            for j in range(len(example_bundles[i])):
                example_bundles[i][j][0]['content'] = example_bundles[i][j][0]['content'].replace("{{shot_number}}", str(i + 1))

        return example_bundles

def few_shot_conversation_prompt_to_jsonl(loaded_prompt: dict, dataset_file: typing.List[str], example_bundles : typing.List[list[dict]], output_file: str):
    populated_prompts = []

    with load_dataset("json", data_files=dataset_file, split="train") as dataset:
        for i, entry in enumerate(dataset):
            # Replace placeholders in the prompt with actual values from the entry
            iter_prompt = replace_placeholders(copy.deepcopy(loaded_prompt)["messages"] if "main" in dataset_file else copy.deepcopy(loaded_prompt)["messages_add"], entry)

            # Replace message with role "Example_1" and "Example_2" with the actual examples from the example bundles
            filtered_iter_prompt = []
            for sub_entry in iter_prompt:
                if "Example" not in sub_entry['role']:
                    filtered_iter_prompt.append(sub_entry)
                else:
                    filtered_iter_prompt += example_bundles[int(sub_entry['role'][-1]) - 1][i%len(example_bundles[0])]

            populated_prompts.append(check_if_train_and_format(filtered_iter_prompt, entry, dataset_file))


    with safe_open_w(output_file) as f_out:
        for item in populated_prompts:
            f_out.write(json.dumps(item) + "\n")

def yaml_prompt_to_jsonl(args : argparse.Namespace):
    loaded_prompt = yaml.safe_load(open(args.prompt, 'r', encoding='utf-8'))

    match loaded_prompt['prompt_type']:
        case '0-shot_conversation':
            zero_shot_conversation_prompt_to_jsonl(loaded_prompt, args.dataset_file, args.output_file)
        case 'few-shot_conversation':
            example_bundles = process_examples(loaded_prompt, args.example_file)
            few_shot_conversation_prompt_to_jsonl(loaded_prompt, args.dataset_file, example_bundles, args.output_file)
        case _:
            raise ValueError(f"Unknown prompt type: {loaded_prompt['prompt_type']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_folder", default="source/", type=str)
    parser.add_argument("--output_folder", default="processed/", type=str)
    parser.add_argument('--prompt', default='../../prompts/OpenBookQA/OpenBookQA_2-shot_llama3.yaml', type=str)
    
    # 'source/add_test.jsonl', 'source/main_dev.jsonl', 'source/add_dev.jsonl']
    parser.add_argument('--dataset_file', default='processed/main-test.jsonl', type=str)
    parser.add_argument('--example_file', default='processed/main-train.jsonl', type=str)
    parser.add_argument('--output_file', default='inference/few-shot/2-shot_main_test.jsonl', type=str)

    parser.add_argument('--task', 
                        choices=['source_to_jsonl', 'source_to_processed_jsonl', 
                                 'yaml_prompt_to_jsonl'], 
                        default='yaml_prompt_to_jsonl', type=str)
    args = parser.parse_args()

    match args.task:
        case 'source_to_jsonl':
            source_to_jsonl(args.source_folder)
        case 'source_to_processed_jsonl':
            source_to_processed_jsonl(args.source_folder, args.output_folder)
        case 'yaml_prompt_to_jsonl':
            yaml_prompt_to_jsonl(args)
        case _:
            raise ValueError(f"Unknown task: {args.task}")


if __name__ == '__main__':
    main()