import csv
from scipy import stats
import numpy as np
import pandas as pd

from automated_llm_eval.prompts import *

def confidence_interval(accuracies, incorrect_samples, correct_samples, confidence_level=0.95):
    """
    Calculate the confidence interval for a list of accuracies.

    Parameters:
    - accuracies: List of accuracy values.
    - sample_size: Number of observations in each accuracy calculation.
    - confidence_level: Desired level of confidence (default is 0.95 for a 95% confidence interval).

    Returns:
    - Tuple containing the lower and upper bounds of the confidence interval.
    """
    print('🥶accuracies are', accuracies)
    mean_accuracy = np.mean(accuracies)
    std_dev = np.std(accuracies, ddof=1)  # ddof=1 for sample standard deviation
    critical_value = stats.norm.ppf((1 + confidence_level) / 2)  # Z-score for normal distribution
    sample_size = len(incorrect_samples)+len(correct_samples)
    margin_of_error = critical_value * (std_dev / np.sqrt(sample_size))

    lower_bound = mean_accuracy - margin_of_error
    upper_bound = mean_accuracy + margin_of_error

    return lower_bound, upper_bound

def get_mode_score_compare():
    df = pd.read_csv("scored_examples/dataset_231103.csv")
    # Calculate the mode of 'q2' for each 'idx' group
    mode_per_idx = df.groupby("idx")["q2"].apply(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    )
    # Create a dictionary mapping 'idx' to the mode of 'q2'
    idx_to_mode = mode_per_idx.to_dict()
    return idx_to_mode


def get_data_split(compare=True, compare_type="iii"):
    train_data = {}
    test_data = {}
    if compare:
        desired_columns = [
            "dataset",
            "idx",
            "q1",
            "q2",
            "q3",
            "q4",
            "inputs",
            "output",
            "target",
            "prompt",
        ]
        with open("scored_examples/dataset_231103.csv", "r") as file:
            # Parse the JSON data and store it as a dictionary
            csv_reader = csv.DictReader(file)
            # Iterate through each row in the CSV
            for line_number, row in enumerate(csv_reader, start=1):
                result = {}
                if type(row) == list or row["dataset"] != compare_type:
                    continue
                for col in desired_columns:
                    result[col] = row[col]
                    if line_number % 5 == 0:
                        test_data[line_number] = result
                    else:
                        train_data[line_number] = result

    else:
        with open("scored_examples/harm_QA.csv", "r") as file:
            desired_columns = ["LLM-Generated Statements", "Human Label (Dev)"]
            csv_reader = csv.DictReader(file)
            # Iterate through each row in the CSV
            for line_number, row in enumerate(csv_reader, start=1):
                result = {}
                for col in desired_columns:
                    result[col] = row[col]
                    if line_number % 5 == 0:
                        test_data[line_number] = result
                    else:
                        train_data[line_number] = result
    return train_data, test_data


def get_policy_file(compare=True):
    if compare:
        with open("policies/summary_compare_correctness_policy.txt", "r") as file:
            current_policy = file.read()
    else:
        with open("policies/safety_policy.txt", "r") as file:
            current_policy = file.read()
    return current_policy


def save_as_csv(data_dict, csv_file):
    with open(csv_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data_dict.keys())
        writer.writeheader()

        for row in zip(*data_dict.values()):
            writer.writerow(dict(zip(data_dict.keys(), row)))
