import pandas as pd
from automated_llm_eval.prompts import *
import csv

def get_mode_score_compare():
    df = pd.read_csv('scored_examples/dataset_231103.csv')
    # Calculate the mode of 'q2' for each 'idx' group
    mode_per_idx = df.groupby('idx')['q2'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    # Create a dictionary mapping 'idx' to the mode of 'q2'
    idx_to_mode = mode_per_idx.to_dict()
    return idx_to_mode

def get_data_split(compare=True, compare_type="iii"):
    train_data= {}
    test_data = {}
    if compare:
        desired_columns = ["dataset", "idx", "q1", "q2", "q3", "q4", "inputs", "output", "target", "prompt"]
        with open('scored_examples/dataset_231103.csv', 'r') as file:
        # Parse the JSON data and store it as a dictionary
            csv_reader = csv.DictReader(file)
            # Iterate through each row in the CSV
            for line_number, row in enumerate(csv_reader, start=1):
                result={}
                if type(row) == list or row["dataset"]!=compare_type:
                    continue
                for col in desired_columns:
                    result[col]=row[col]
                    if line_number%5==0:
                        test_data[line_number]=result
                    else:
                        train_data[line_number] = result
                
    else:
        with open('scored_examples/harm_QA.csv', 'r') as file:
            desired_columns = ["LLM-Generated Statements", "Human Label (Dev)"]
            csv_reader = csv.DictReader(file)
            # Iterate through each row in the CSV
            for line_number, row in enumerate(csv_reader, start=1):
                result={}
                for col in desired_columns:
                    result[col]=row[col]
                    if line_number%5==0:
                        test_data[line_number]=result
                    else:
                        train_data[line_number] = result
    return train_data, test_data

def get_policy_file(compare=True):
    if compare:
        with open('policies/summary_compare_correctness_policy.txt', 'r') as file:
            current_policy = file.read()
    else:
        with open('policies/safety_policy.txt', 'r') as file:
            current_policy = file.read()
    return current_policy

def save_as_csv(data_dict, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data_dict.keys())
        writer.writeheader()
        
        for row in zip(*data_dict.values()):
            writer.writerow(dict(zip(data_dict.keys(), row)))