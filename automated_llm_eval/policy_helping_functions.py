import csv
from scipy import stats
import numpy as np
import pandas as pd
import difflib
from IPython.display import display, HTML
import math
import random

from automated_llm_eval.prompts import *
from automated_llm_eval.accuracy_metrics import AccuracyMetrics

def fleiss_kappa(mat):
    """ Computes the Kappa value
        @param n Number of rating per subjects (number of human raters)
        @param mat Matrix[subjects][categories]
        @return The Kappa value """

    def checkEachLineCount(mat):
        """ Assert that each line has a constant number of ratings
            @param mat The matrix checked
            @return The number of ratings
            @throws AssertionError If lines contain different number of ratings """
        n = sum(mat[0])
        
        assert all(sum(line) == n for line in mat[1:]), "Line count != %d (n value)." % n
        return n

    n = checkEachLineCount(mat)   # PRE : every line count must be equal to n
    N = len(mat)
    k = len(mat[0])
    
    # Computing p[]
    p = [0.0] * k
    for j in range(k):
        p[j] = 0.0
        for i in range(N):
            p[j] += mat[i][j]
        p[j] /= N*n
    
    # Computing P[]    
    P = [0.0] * N
    for i in range(N):
        P[i] = 0.0
        for j in range(k):
            P[i] += mat[i][j] * mat[i][j]
        P[i] = (P[i] - n) / (n * (n - 1))
    # if DEBUG: print "P =", P
    
    # Computing Pbar
    Pbar = sum(P) / N
    # if DEBUG: print "Pbar =", Pbar
    
    # Computing PbarE
    PbarE = 0.0
    for pj in p:
        PbarE += pj * pj
    try:
        kappa = (Pbar - PbarE) / (1 - PbarE)
    except:
        kappa=1
    return kappa



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
    mean_accuracy = np.mean(accuracies)
    std_dev = np.std(accuracies, ddof=1)  # ddof=1 for sample standard deviation
    critical_value = stats.norm.ppf((1 + confidence_level) / 2)  # Z-score for normal distribution
    sample_size = len(incorrect_samples)+len(correct_samples)
    margin_of_error = critical_value * (std_dev / np.sqrt(sample_size))

    lower_bound = mean_accuracy - margin_of_error
    upper_bound = mean_accuracy + margin_of_error

    return lower_bound, upper_bound

def get_mode_score_compare():
    df = pd.read_csv("scored_examples/dataset_231103.csv", engine= "python")
    # Calculate the mode of 'q2' for each 'idx' group
    mode_per_idx = df.groupby("idx")["q2"].apply(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    )
    # Create a dictionary mapping 'idx' to the mode of 'q2'
    idx_to_mode = mode_per_idx.to_dict()
    return idx_to_mode

def add_fleiss_column():
     df = pd.read_csv("scored_examples/dataset_231103.csv")
     index_to_scores = calculate_fleiss_kappa("scored_examples/dataset_231103.csv")
     df['Scores'] = df['idx'].map(index_to_scores)
     df.to_csv("VanDeen_updated.csv", index=False)

def calculate_fleiss_kappa(dataset):
    """
    Calculate Fleiss' Kappa for each text corpus.

    Parameters:
    - dataframe: Pandas DataFrame containing the data.
    - column_name: Name of the column containing the ratings.

    Returns:
    - Dictionary mapping text corpus index to Fleiss' Kappa value.
    """

    def aggregate_for_fleiss(list_scores):
        return_table = []
        for score in list_scores:
            if score == -2:
                temp=[1,0,0,0,0]
            elif score ==-1:
                temp=[0,1,0,0,0]
            elif score == 0:
                temp=[0,0,1,0,0]
            elif score == 1:
                temp=[0,0,0,1,0]
            else:
                temp=[0,0,0,0,1]
            return_table.append(temp)
        return [[sum(x) for x in zip(*return_table)]]

    # Pivot the DataFrame to get a matrix of ratings
    df = pd.read_csv(dataset)
    pivot_df = df.pivot(index='reader', columns='idx', values='q2')
    # Calculate Fleiss' Kappa for each text corpus
    kappa_values = {}
    for idx, column_data in pivot_df.items():
        # Extract ratings for the current text corpus
        ratingsList = list(column_data.values)
        filtered_list = [value for value in ratingsList if not math.isnan(value)]
        agg = aggregate_for_fleiss(filtered_list)
        fleiss_kappa_res = fleiss_kappa(agg)
        kappa_values[idx] = fleiss_kappa_res

    return kappa_values

def get_data_split(compare=True, compare_type="iii", reliability_type="high"):
    train_data = {}
    test_data= {}
    high_reliable_data = {}
    medium_reliable_data = {}
    low_reliable_data = {}
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
        with open("scored_examples/VanDeen_updated.csv", "r") as file:
            # Parse the JSON data and store it as a dictionary
            csv_reader = csv.DictReader(file)
            # Iterate through each row in the CSV
            for line_number, row in enumerate(csv_reader, start=1):
                result = {}
                if type(row) == list or row["dataset"] != compare_type:
                    continue
                fleiss_score = row["Scores"]
                try:
                    fleiss_score=int(fleiss_score)
                except:
                    fleiss_score=round(float(fleiss_score), 2)
                if fleiss_score>-.26: #chose -.26 because there are three potential fleiss scores and this chooses top two as high reliability
                    for col in desired_columns:
                        result[col] = row[col]
                        high_reliable_data[line_number] = result
                # if fleiss_score==-.25:
                #     for col in desired_columns:
                #         result[col] = row[col]
                #         medium_reliable_data[line_number] = result
                else:
                    for col in desired_columns:
                        result[col] = row[col]
                        low_reliable_data[line_number] = result
        keys = list(high_reliable_data.keys())
        random.shuffle(keys)

        split_point = int(len(keys) * .8)
        # print([high_reliable_data[key]['q2'] for key in keys])
        # print(len(set([high_reliable_data[key]['idx'] for key in keys])))
        train_data = {key: high_reliable_data[key] for key in keys[:split_point]}
        high_reliable_data_test = {key: high_reliable_data[key] for key in keys[split_point:]}
        if reliability_type == "high":
            test_data = high_reliable_data_test
        # elif reliability_type == "medium":
        #     test_data = medium_reliable_data
        elif reliability_type == "low":
            test_data = low_reliable_data
        return train_data, test_data

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

def compute_metrics(accuracy_metrics_object: AccuracyMetrics):
    accuracy = accuracy_metrics_object.compute_accuracy()
    f1 = accuracy_metrics_object.compute_f1_score()
    precision = accuracy_metrics_object.compute_precision()
    recall = accuracy_metrics_object.compute_recall()
    incorrect_COT, correct_COT = accuracy_metrics_object.get_COT()
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall, "COT": [incorrect_COT, correct_COT]}

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

    if diff_exists:
        display(HTML(diff_table))
    else:
        print("No differences found.")