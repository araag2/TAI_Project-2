import argparse
import typing
import yaml
import json
import copy

from datasets import load_dataset

def source_to_jsonl(source_path: str = "source/"):
    main_train = load_dataset("parquet", data_files=f"{source_path}main_train.parquet", split="train")
    add_train = load_dataset("parquet", data_files=f"{source_path}add_train.parquet", split="train")

    main_test = load_dataset("parquet", data_files=f"{source_path}main_test.parquet", split="train")
    add_test = load_dataset("parquet", data_files=f"{source_path}add_test.parquet", split="train")

    main_dev = load_dataset("parquet", data_files=f"{source_path}main_dev.parquet", split="train")
    add_dev = load_dataset("parquet", data_files=f"{source_path}add_dev.parquet", split="train")

    #output_to_jsonl
    main_train.to_json(source_path + "main_train.jsonl")
    add_train.to_json(source_path + "add_train.jsonl")

    main_test.to_json(source_path + "main_test.jsonl")
    add_test.to_json(source_path + "add_test.jsonl")

    main_dev.to_json(source_path + "main_dev.jsonl")
    add_dev.to_json(source_path + "add_dev.jsonl")

key_to_option = {0 : "A", 1 : "B", 2 : "C", 3 : "D"}

def source_to_processed_jsonl(source_path: str = "source/", output_path: str = "processed/"):
    main_train = load_dataset("parquet", data_files=f"{source_path}main_train.parquet", split="train")
    add_train = load_dataset("parquet", data_files=f"{source_path}add_train.parquet", split="train")

    main_test = load_dataset("parquet", data_files=f"{source_path}main_test.parquet", split="train")
    add_test = load_dataset("parquet", data_files=f"{source_path}add_test.parquet", split="train")

    main_dev = load_dataset("parquet", data_files=f"{source_path}main_dev.parquet", split="train")
    add_dev = load_dataset("parquet", data_files=f"{source_path}add_dev.parquet", split="train")

     # Columns names in the dataset [id, question_stem, choices [text, labels], answerKey] and ADD_ONLY [fact1, humanScore, clarity, turkIdAnonymized]

    for name, dataset in zip(["main_train", "add_train", "main_test", "add_test", "main_dev", "add_dev"],
                             [main_train, add_train, main_test, add_test, main_dev, add_dev]):
        
        dataset = dataset.rename_column("question_stem", "Question")
        dataset = dataset.rename_column("answerKey", "Label")

        # Go into column Choices, and divide text into 4 fields: Text_Answer_1, Text_Answer_2, Text_Answer_3, Text_Answer_4
        choices = dataset["choices"]
        for i in range(4):
            dataset = dataset.add_column(f"Option_{key_to_option[i]}", [choice["text"][i] for choice in choices])
        dataset = dataset.remove_columns(["choices"])

        if "add" in name:
            dataset = dataset.rename_column("fact1", "Supporting_Fact")
            dataset = dataset.rename_column("humanScore", "Human_Score")
            dataset = dataset.rename_column("clarity", "Clarity_Score")
            dataset = dataset.remove_columns(["turkIdAnonymized"])

        dataset.to_json(f"{output_path}{name}.jsonl", lines=True)

def zero_shot_conversation_prompt_to_jsonl(loaded_prompt : dict, dataset_file : typing.List[str], output_file : str):
    with load_dataset("json", data_files=dataset_file, split="train") as dataset:
        # Iterate over the dataset and create a new dictionary with the prompt format
        populated_prompts = []

        for entry in dataset:
            iter_prompt = copy.deepcopy(loaded_prompt)["messages"] if "main" in dataset_file[0] else copy.deepcopy(loaded_prompt)["messages_add"]

            for message in iter_prompt:
                if message['content']:
                    for key, value in entry.items():
                        message['content'] = message['content'].replace(f"{{{{{key}}}}}", str(value))
                

            # If example is not training, delete the last assistant message, because we don't want to prompt the model with an answer
            if "train" not in dataset_file[0] and iter_prompt[-1]['role'] == 'assistant':
                iter_prompt.pop(-1)

            populated_prompts.append({"id" : entry["id"], "messages" : iter_prompt, "Label" : entry["Label"]})

    # Write the populated prompts to the output file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for prompt in populated_prompts:
            f_out.write(json.dumps(prompt) + '\n')


def yaml_prompt_to_jsonl(prompt_file: str, dataset_file: typing.List[str], output_file: str):
    loaded_prompt = yaml.safe_load(open(prompt_file, 'r', encoding='utf-8'))

    match loaded_prompt['prompt_type']:
        case '0-shot_conversation':
            zero_shot_conversation_prompt_to_jsonl(loaded_prompt, dataset_file, output_file)
        case _:
            raise ValueError(f"Unknown prompt type: {loaded_prompt['prompt_type']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_folder", default="source/", type=str)
    parser.add_argument("--output_folder", default="processed/", type=str)
    parser.add_argument('--prompt', default='../../prompts/OpenBookQA/OpenBookQA_0-shot_llama3.yaml', type=str)
    
    # 'source/add_test.jsonl', 'source/main_dev.jsonl', 'source/add_dev.jsonl']
    parser.add_argument('--dataset_file', default='processed/main_test.jsonl', type=str)
    parser.add_argument('--output_file', default='inference/0-shot/0-shot_main_test.jsonl', type=str)

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
            yaml_prompt_to_jsonl(args.prompt, args.dataset_file, args.output_file)
        case _:
            raise ValueError(f"Unknown task: {args.task}")


if __name__ == '__main__':
    main()