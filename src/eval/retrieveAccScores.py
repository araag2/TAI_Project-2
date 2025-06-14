import re
import glob
import csv
import argparse
from collections import defaultdict

def extract_info_from_md(file_path):
    json_file_name = None
    accuracy = None

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

        # Extract JSONL file name
        file_name_match = re.search(r'File name:\s*(.*?)(?:\s|$)', content)
        if file_name_match:
            file_path_str = file_name_match.group(1).strip()
            json_file_name = file_path_str.split('/')[-1]

        # Extract Accuracy
        acc_match = re.search(r'Accuracy\s*-\s*([\d.]+)', content)
        if acc_match:
            accuracy = acc_match.group(1)

    return json_file_name, accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_dir', type=str, help='top directory to search for markdown files', default='outputs/')
    parser.add_argument('--output_dir', type=str, help='path to output path', default='')
    args = parser.parse_args()

    md_files = md_files = glob.glob(f'{args.top_dir}**/*.md', recursive=True)

    # Data structure: { model_name: {set_type: accuracy} }
    results = defaultdict(dict)

    for md_file in md_files:
        json_file_name, accuracy = extract_info_from_md(md_file)
        if not json_file_name or not accuracy:
            print(f"⚠️ Could not extract info from {md_file}")
            continue

        parts = json_file_name.replace('.jsonl', '').split('_')

        # Get set type (last part)
        set_type = parts[-1].lower()

        # Remove dataset and set type to get model key
        model_key_parts = [p for p in parts if p != 'OpenBookQA'][:-1]  # drop last (set type)
        model_key = '_'.join(model_key_parts)

        results[model_key][set_type] = accuracy

    # Write to CSV
    with open('summary.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'file_name',
            'Accuracy Main-Dev',
            'Accuracy Main-Test',
            'Accuracy Add-Dev',
            'Accuracy Add-Test'
        ])

        for model_key in sorted(results.keys()):
            acc_dict = results[model_key]
            writer.writerow([
                model_key,
                acc_dict.get('main-dev', ''),
                acc_dict.get('main-test', ''),
                acc_dict.get('add-dev', ''),
                acc_dict.get('add-test', '')
            ])

    print("✅ summary.csv written.")

    with open('summary_table_rows.txt', 'w', encoding='utf-8') as table_file:
        for model_key in sorted(results.keys()):
            acc_dict = results[model_key]

            # Clean model key if needed (e.g., remove no-training_0-shot_)
            # This assumes you want to keep CoT_gemma-4B etc.
            cleaned_key = model_key
            cleaned_key = cleaned_key.replace('no-training_', '')
            cleaned_key = cleaned_key.replace('0-shot_', '')

            row = f"{cleaned_key} & " \
                f"{acc_dict.get('main-dev', '')} & " \
                f"{acc_dict.get('add-dev', '')} & " \
                f"{acc_dict.get('main-test', '')} & " \
                f"{acc_dict.get('add-test', '')} \\\\"
            table_file.write(row + '\n')

    print("✅ summary_table_rows.txt written.")

if __name__ == "__main__":
    main()