import csv
import pandas as pd
import difflib
import random

from automated_llm_eval.prompts import *
from automated_llm_eval.accuracy_metrics import AccuracyMetrics

def get_mode_score_compare():
    print('STASRT INDEXING')
    df = pd.read_csv('scored_examples/VanDeenCollapsed_updated.csv', engine= "python")
    # Calculate the mode of 'q2' for each 'idx' group
    mode_per_idx = df.groupby("idx")["q2"].apply(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    )
    # Create a dictionary mapping 'idx' to the mode of 'q2'
    idx_to_mode = mode_per_idx.to_dict()
    print('ENDING INDEX')
    return idx_to_mode

def get_data_split(task:str, compare_type, reliability_type):
    train_data = {}
    test_data= {}
    high_reliable_data = {}
    low_reliable_data = {}
    if task=='compare':
        desired_columns = [
            "dataset",
            "idx",
            "q1",
            "q2",
            "q3",
            "q4",
            "inputs",
            "output",
            "target"
        ]
        with open('scored_examples/VanDeenCollapsed_updated.csv', "r") as file:
            # Parse the JSON data and store it as a dictionary
            csv_reader = csv.DictReader(file)
            # Iterate through each row in the CSV
            for line_number, row in enumerate(csv_reader, start=1):
                result = {"id": line_number}
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
            
        unique_values_count = {}
        for inner_dict in test_data.values():
            if 'q2' in inner_dict:
                q2_value = inner_dict['q2']
                unique_values_count[q2_value] = unique_values_count.get(q2_value, 0) + 1

        # Print unique values and their occurrences
        for value, count in unique_values_count.items():
            print(f"Value: {value}, Occurrences: {count}")
        
        return train_data, test_data

    elif task=="qa":
        with open("scored_examples/redteaming.csv", "r") as file:
            desired_columns = ["Label", "Response", "Prompt text"]
            csv_reader = csv.DictReader(file)
            filtered_rows = [row for row in csv_reader if row["Label"] != "Unsure"]
            for line_number, row in enumerate(filtered_rows, start=1):
                if line_number<2:
                    continue
                result = {"id": line_number}
                for col in desired_columns:
                    if row[col]=="Appropriate":
                        result[col] = 1
                    elif row[col]=="Inappropriate":
                        result[col]=0
                    else:
                        result[col] = row[col]
                    if line_number % 5 == 0:
                        test_data[line_number] = result
                    else:
                        train_data[line_number] = result
    elif task=="harm":
        with open("scored_examples/harm.csv", "r") as file:
            desired_columns = ["Correct Answer", "Question", "Erroneous Answer", "Final label"]
            csv_reader = csv.DictReader(file)
            for line_number, row in enumerate(csv_reader, start=1):
                if line_number<1:
                    continue
                result = {"id": line_number}
                for col in desired_columns:
                    result[col] = row[col]
                    if line_number % 5 == 0:
                        test_data[line_number] = result
                    else:
                        train_data[line_number] = result
        print('train data is', train_data)
    return train_data, test_data


def get_policy_file(task:str):
    if task=='compare':
        with open("policies/summary_compare_correctness_policy.txt", "r") as file:
            current_policy = file.read()
    elif task == 'qa':
        with open("policies/redteaming_policy.txt", "r") as file:
            current_policy = file.read()
    elif task == 'harm':
        with open("policies/safety_policy.txt", "r") as file:
            current_policy = file.read()
    return current_policy