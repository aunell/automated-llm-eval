import pandas as pd
from tqdm import tqdm
from prompts import *
from create_chat_completion import create_chat_completion
import csv

def save_as_csv(data_dict, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data_dict.keys())
        writer.writeheader()
        
        for row in zip(*data_dict.values()):
            writer.writerow(dict(zip(data_dict.keys(), row)))

def create_agent_response(openai_token, current_policy, source_text, compare, statement_a=None, statement_b=None):
    gpt_system_prompt = "You are an expert AI agent, possessing an in-depth knowledge and expertise within the healthcare and medical domain."
    if compare:
        compare_gpt_prompt = COMPARE_AGENT_PROMPT.format(source = source_text, current_policy=current_policy, summary_a = statement_a, summary_b = statement_b)
        gpt_response, _ = create_chat_completion("gpt-4", 
                                                            gpt_system_prompt, 
                                                            compare_gpt_prompt, 
                                                            openai_token)   
    else:
        safety_gpt_prompt = QA_AGENT_PROMPT.format(statement=source_text, current_policy=current_policy)
        gpt_response, _ = create_chat_completion("gpt-4", 
                                                        gpt_system_prompt, 
                                                        safety_gpt_prompt, 
                                                        openai_token)
    return gpt_response    

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
                # if type(row) == list or row["dataset"]!=compare_type:
                #     continue
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

def check_policy_accuracy(dataset, openai_token, current_policy, compare):
    incorrect_labelled = []
    correct_labelled=[]
    if compare:
        idx_to_mode = get_mode_score_compare()
    print('length is', len(dataset.values()))
    k=0
    for example in dataset.values():
        print('WE ARE ON THE ', k, 'th example')
        if k>10:
            break
        if compare:    
            print('compare', compare)
            human_score = idx_to_mode[int(example["idx"])]
            statement = example["inputs"]
            agent_response = example["output"]
            human_response = example["target"]
            agent_response_score = create_agent_response(openai_token, current_policy, statement, compare, human_response, agent_response)
        else:
            statement = example["LLM-Generated Statements"]
            human_score = int(example["Human Label (Dev)"])
            agent_response_score = create_agent_response(openai_token, current_policy, statement, compare)

        SCORE_RETRIEVAL = SCORE_RETRIEVAL_PROMPT.format(response=agent_response_score) 
        response_score_string, _ = create_chat_completion("gpt-4",   score_retrieval_character_prompt, SCORE_RETRIEVAL, openai_token) 
        try:
            response_score = int(response_score_string)
            if response_score != human_score:
                print('incorrect result from agent')
                statement_analysis = "The following statement: " + statement + " was given a score of: " + str(response_score) +" by the agent, but the correct score should have been: " + str(human_score) + ". The agent's reasoning for this score is as follows: " + agent_response_score
                print(statement_analysis)
                incorrect_labelled.append(statement_analysis)
            else:
                # statement_analysis = "The following statement was judged correctly by the agent: " + statement + "The agent's reasoning for this score is as follows: " + agent_response
                # print(statement_analysis)
                correct_labelled.append(statement)
        except:
            print('RESPONSE SCORE STRING ERROR:', response_score_string)
        k+=1
    score = len(correct_labelled)/ (len(correct_labelled)+len(incorrect_labelled))  
    print('score is, ', score)
    return current_policy, score, correct_labelled, incorrect_labelled
        
def policy_tuning(agent, openai_token, output, compare, compare_type='iii'):
    score = 0.0
    train_data, test_data = get_data_split(compare, compare_type)
    current_policy = get_policy_file(compare)

    print('checking acc')
    save_as_csv(train_data, f"{compare_type}_train.csv")
    save_as_csv(test_data, f"{compare_type}_test.csv")
    current_policy_before, score_before, _, _ = check_policy_accuracy(test_data, openai_token, current_policy, compare)
    print('done check')
    data = {}
    i=0
    
    while score <.9 and i<3:
        print('score is', score)
        current_policy, score, correct_labelled, incorrect_labelled = check_policy_accuracy(train_data, openai_token, current_policy, compare)
        data[i]=[current_policy, score, correct_labelled, incorrect_labelled]
        AGENT_IMPROVEMENT = POLICY_MUTATE_PROMPT_TEMPLATE.format(original_policy = current_policy, correct_answers = correct_labelled, incorrect_answers = incorrect_labelled)
        current_policy, _ =  create_chat_completion(agent, prompt_improvement_character_prompt, AGENT_IMPROVEMENT, openai_token)
        print('NEW POLICY for i IS:', i, current_policy)
        save_as_csv(data, 'policy_mutation_snapshot.csv')
        i+=1

    current_policy_after, score_after, _, _, = check_policy_accuracy(test_data, openai_token, current_policy, compare)
    data["final scores"] = [score_before, score_after]
    save_as_csv(data, output)
    return current_policy