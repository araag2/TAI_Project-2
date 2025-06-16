import json

def replace_completion_content(original_file, reasoning_file, output_file):
    with open(original_file, 'r', encoding='utf-8') as f_orig, \
         open(reasoning_file, 'r', encoding='utf-8') as f_reason, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for orig_line, reason_line in zip(f_orig, f_reason):
            orig = json.loads(orig_line)
            reason = json.loads(reason_line)
            
            # Replace completion content
            orig["completion"][0]["content"] = reason["Answer"]
            
            # Write updated line
            f_out.write(json.dumps(orig, ensure_ascii=False) + "\n")

# Example usage
replace_completion_content(
    "data/OpenBookQA/inference/CoT-0-shot/CoT-0-shot_add-train.jsonl", 
    "outputs/no_training/OpenBookQA/Qwen-8B/Reasoning/Reasoning_8-bit_OpenBookQA_Qwen-14B_add-train.jsonl", 
    "data/OpenBookQA/inference/Reasoning/Train-Reasoning-CoT-Qwen3_add-train.jsonl")