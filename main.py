from get_questions import get_questions
from prompts import *
from private_key import *
from test import run_test
from model_analysis import analysis
from visualize import *
from policy_tuning import *

import sys
   
openai_token = key["open-ai"]

def general_response_experiment():
    engine_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]
    judge_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]

    run_test(engine_options, judge_options)

    for engine in engine_options:
        for engine_judge in judge_options:
            analysis(engine, engine_judge, engine+ ' + ' +engine_judge)

    create_plots(engine_options, judge_options)

# policy_tuning("gpt-4", openai_token, 'policy_mutation_track_neg_pls.csv', compare=True, compare_type = 'pls')
# create_accuracy_plot('policy_mutation_track_neg_pls.csv', "Accuracy of Policy by Iteration: Negative COT", "acc_policy_neg_COT.png")
# create_len_of_policy_plot('policy_mutation_track_neg_pls.csv', "Length of Policy by Iteration: Negative COT", "len_policy_neg_COT.png")

policy_tuning("gpt-4", openai_token, 'policy_mutation_track_neg_pls.csv', compare=False, compare_type = 'pls')
create_accuracy_plot('policy_mutation_QA_neg.csv', "Accuracy of Policy by Iteration: Negative COT", "acc_policy_neg_COT_QA.png")
create_len_of_policy_plot('policy_mutation_QA_neg.csv', "Length of Policy by Iteration: Negative COT", "len_policy_neg_COT_QA.png")