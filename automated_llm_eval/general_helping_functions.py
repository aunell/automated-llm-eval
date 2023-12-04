import csv
import pandas as pd
import difflib
import random

from automated_llm_eval.prompts import *
from automated_llm_eval.accuracy_metrics import AccuracyMetrics

def select_batch(dataset: dict, batch_size: int, seed: int = 42) -> list:
    examples = list(dataset.values())
    random.Random(seed).shuffle(examples)
    batch = examples[: len(examples) // batch_size]
    return batch

def save_dict_as_csv(my_dict, file_path):
    # Extract row labels and column labels
    row_labels = list(my_dict[0].keys())
    col_labels = list(my_dict.keys())

    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=[''] + col_labels)

        # Write header row with keys as column names
        csv_writer.writeheader()

        # Write data rows
        for row_label in row_labels:
            row_data = {'': row_label}
            for col_label in col_labels:
                row_data[col_label] = my_dict[col_label][row_label]
            csv_writer.writerow(row_data)

def compute_metrics(accuracy_metrics_object: AccuracyMetrics):
    accuracy = accuracy_metrics_object.compute_accuracy()
    f1 = accuracy_metrics_object.compute_f1_score()
    precision = accuracy_metrics_object.compute_precision()
    recall = accuracy_metrics_object.compute_recall()
    incorrect_COT, correct_COT = accuracy_metrics_object.get_COT()
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall, "COT": [incorrect_COT, correct_COT]}

def editDistance(previous_response: str, response: str):
    d = difflib.Differ()
    diff = d.compare(previous_response.splitlines(), response.splitlines())
    diff_list = list(diff)
    added_count = sum(len(item) for item in diff_list if item.startswith('+'))
    deleted_count = sum(len(item) for item in diff_list if item.startswith('-'))
    return added_count+deleted_count

def compare_responses(previous_response: str, response: str):
    d = difflib.Differ()
    diff = d.compare(previous_response.splitlines(), response.splitlines())

    diff_table = "<table>"
    diff_exists = False

    for line in diff:
        if line.startswith("- "):
            diff_table += f"<tr style='color: red;'><td>{line}</td></tr>"
            diff_exists = True
        elif line.startswith("+ "):
            diff_table += f"<tr style='color: green;'><td>{line}</td></tr>"
            diff_exists = True
        else:
            diff_table += f"<tr><td>{line}</td></tr>"

    diff_table += "</table>"

    return diff_table

def find_average(lst):
    if not lst:
        return None  # Handle empty list case to avoid division by zero

    total_sum = sum(lst)
    average = total_sum / len(lst)
    return average  