#!/usr/bin/env python3

import json
import os
import os.path
import warnings
import argparse
import random

from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

warnings.simplefilter('ignore')

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')


def F1_Recall_Precision(gold_labels, pred_labels, average):
    F1 = f1_score(gold_labels, pred_labels, average=average)
    Recall = recall_score(gold_labels, pred_labels, average=average)
    Precision = precision_score(gold_labels, pred_labels, average=average)
    return F1, Recall, Precision


def calc_scores(prediction_file : str, gold_file : str, output_dir : str):
    preds = load_dataset('json', data_files=prediction_file, split='train')
    gold = load_dataset('json', data_files=gold_file, split='train')

    # Each answer should be in the form A,B,C or D
    valid_answers = {"A", "B", "C", "D"}

    def check_valid_answers(answer):
        if answer not in valid_answers:
            #raise ValueError(f"Answer '{answer}' not in Expected Format. Expected one of {valid_answers}.")
            return random.choice(list(valid_answers)) # Default to random answer if invalid
        return answer 

    pred_labels = [check_valid_answers(example["Answer"][-1]) for example in preds]
    gold_labels = [check_valid_answers(example["Label"][-1]) for example in gold]


    # Metrics
    F1_micro, Rec_micro, Prec_micro = F1_Recall_Precision(gold_labels, pred_labels, "micro")
    F1_macro, Rec_macro, Prec_macro = F1_Recall_Precision(gold_labels, pred_labels, "macro")
    F1_weighted, Rec_weighted, Prec_weighted = F1_Recall_Precision(gold_labels, pred_labels, "weighted")

    with safe_open_w(f'{output_dir}SCORES-{prediction_file.split("/")[-1][:-6]}.md') as f:
        print(f'# Full Evaluation Scores\n', file=f)
        print(f'File name: {prediction_file}\n', file=f)

        print(f'\n---\n', file=f)

        print(f'## Leaderboard Scores\n', file=f)

        #print(f'Metrics (%): Accuracy - {Accuracy*100:.1f}', file=f)

        print(f'Metrics (%): F1-Score-Micro | Recall-Micro | Precision-Micro | Average-Micro', file=f)
        print(f'                {F1_micro*100:.1f}        {Rec_micro*100:.1f}          {Prec_micro*100:.1f}        {(F1_micro + Rec_micro + Prec_micro) / 3 * 100:.1f}', file=f)

        print(f'Metrics (%): F1-Score-Macro | Recall-Macro | Precision-Macro | Average-Macro', file=f)
        print(f'                {F1_macro*100:.1f}        {Rec_macro*100:.1f}          {Prec_macro*100:.1f}        {(F1_macro + Rec_macro + Prec_macro) / 3 * 100:.1f}', file=f)

        print(f'Metrics (%): F1-Score-Weighted | Recall-Weighted | Precision-Weighted | Average-Weighted', file=f)
        print(f'                {F1_weighted*100:.1f}        {Rec_weighted*100:.1f}          {Prec_weighted*100:.1f}        {(F1_weighted + Rec_weighted + Prec_weighted) / 3 * 100:.1f}', file=f)

        print(f'\n---\n', file=f)

        #if args != None:
        #    print(f'## Full Arg List\n', file=f)
        #    for arg in vars(args):
        #        print(f'{arg} = {getattr(args, arg)}', file=f)

        print(f'\n---\n', file=f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--prediction_file', type=str, help='path to input res file with labels for each query', default='../../outputs/no_training/OpenBookQA/llama3/0-shot/no-training_0-shot_OpenBookQA_llama3_add-dev.jsonl')
    parser.add_argument('--gold_file', type=str, help='path to input res file with labels for each query', default='')
    parser.add_argument('--output_dir', type=str, help='path to output file with scores', default='')
    args = parser.parse_args()

    calc_scores(args.prediction_file, args.gold_file, args.output_dir)


if '__main__' == __name__:
    main()