import pandas as pd
from tqdm import tqdm
from automated_llm_eval.prompts import *
from automated_llm_eval.create_chat_completion import create_chat_completion
from automated_llm_eval.chat_model import ChatModel
from automated_llm_eval.utils import ProgressBar
import random
from automated_llm_eval.policy_helping_functions import *

def create_agent_response(current_policy, source_text, compare, statement_a=None, statement_b=None):
    gpt_system_prompt = "You are an expert AI agent, possessing an in-depth knowledge and expertise within the healthcare and medical domain."
    model = ChatModel(model="gpt-3.5-turbo-1106", temperature=0.5, top_p=0.5, max_tokens=700, seed=42)
    if compare:
        compare_gpt_prompt = COMPARE_AGENT_PROMPT.format(source = source_text, current_policy=current_policy, summary_a = statement_a, summary_b = statement_b)
        gpt_response = model.create_chat_completion(gpt_system_prompt, compare_gpt_prompt, output_format='simple')
    else:
        safety_gpt_prompt = QA_AGENT_PROMPT.format(statement=source_text, current_policy=current_policy)
        gpt_response = model.create_chat_completion(gpt_system_prompt, safety_gpt_prompt, output_format='simple')
    return gpt_response    

def check_policy_accuracy(dataset, current_policy, batchsize, compare):
    incorrect_labelled = []
    correct_labelled=[]
    if compare:
        idx_to_mode = get_mode_score_compare()
    k=0
    shuffled_data= list(dataset.values())
    random.shuffle(shuffled_data)
    for example in shuffled_data[:len(shuffled_data)//batchsize]:
        print('WE ARE ON THE ', k, 'th example')
        if compare:    
            human_score = idx_to_mode[int(example["idx"])]
            statement = example["inputs"]
            agent_response = example["output"]
            human_response = example["target"]
            agent_response_score = create_agent_response(current_policy, statement, compare, human_response, agent_response)
        else:
            statement = example["LLM-Generated Statements"]
            human_score = int(example["Human Label (Dev)"])
            agent_response_score = create_agent_response(current_policy, statement, compare)

        SCORE_RETRIEVAL = SCORE_RETRIEVAL_PROMPT.format(response=agent_response_score) 
        model_score = ChatModel(model="gpt-3.5-turbo-1106", temperature=.1, top_p=0.5, max_tokens=700, seed=42)
        response_score_string = model_score.create_chat_completion(score_retrieval_character_prompt, SCORE_RETRIEVAL, output_format='simple')
        try:
            response_score = int(response_score_string)
            if response_score != human_score:
                print('incorrect result from agent')
                statement_analysis = "The following statement: " + statement + " was given a score of: " + str(response_score) +" by the agent, but the correct score should have been: " + str(human_score) + ". The agent's reasoning for this score is as follows: " + agent_response_score
                incorrect_labelled.append(statement_analysis)
            else:
                correct_labelled.append(statement)
        except:
            print('RESPONSE SCORE STRING ERROR:', response_score_string)
        k+=1
    score = len(correct_labelled)/ (len(correct_labelled)+len(incorrect_labelled)) or 0
    return current_policy, score, correct_labelled, incorrect_labelled
        
def policy_tuning(output, compare, batch_size, compare_type='iii'):
    score = 0.0
    train_data, test_data = get_data_split(compare, compare_type)
    current_policy = get_policy_file(compare)
    _, score_before, _, _ = check_policy_accuracy(test_data, current_policy, batch_size, compare)
    data = {}
    i=0

    while score <.9 and i<10:
        print('score is', score)
        current_policy, score, correct_labelled, incorrect_labelled = check_policy_accuracy(train_data, current_policy, batch_size, compare)
        data[i]=[current_policy, score, correct_labelled, incorrect_labelled]
        AGENT_IMPROVEMENT = POLICY_MUTATE_PROMPT_TEMPLATE.format(original_policy = current_policy, correct_answers = correct_labelled, incorrect_answers = incorrect_labelled)
        model = ChatModel(model="gpt-3.5-turbo-1106", temperature=0.5, top_p=0.5, max_tokens=700, seed=42)
        try:
            current_policyNew= model.create_chat_completion(prompt_improvement_character_prompt, AGENT_IMPROVEMENT, output_format="simple")
            if current_policyNew!= None:
                current_policy=current_policyNew
        except:
            pass
        save_as_csv(data, f"results/csv/policy_mutation_snapshot_{compare_type}_compare{compare}.csv")
        i+=1

    _, score_after, _, _, = check_policy_accuracy(test_data, current_policy, batch_size, compare)
    data["final scores"] = [score_before, score_after]
    save_as_csv(data, output)
    return current_policy