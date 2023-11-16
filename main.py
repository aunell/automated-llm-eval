from automated_llm_eval.get_questions import get_questions
from automated_llm_eval.prompts import *
from private_key import *
from automated_llm_eval.test import run_test
from automated_llm_eval.model_analysis import analysis
from automated_llm_eval.visualize import *
from automated_llm_eval.policy_tuning import *

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

def run_compare(compare_type):
    policy_tuning(f"results/csv/policy_mutation_track_neg_{compare_type}.csv", compare=True, batch_size = 10, compare_type=compare_type)
    create_accuracy_plot(f"results/csv/policy_mutation_track_neg_{compare_type}.csv", "Accuracy of Policy by Iteration: Negative COT", f"results/visualizations/acc_policy_neg_COT_{compare_type}.png")
    create_len_of_policy_plot(f"results/csv/policy_mutation_track_neg_{compare_type}.csv", "Length of Policy by Iteration: Negative COT", f"results/visualizations/len_policy_neg_COT_{compare_type}.png")

def run_QA():
    policy_tuning('results/csv/policy_mutation_QA_neg.csv', compare=False, batch_size = 1, compare_type = 'pls')
    create_accuracy_plot('results/csv/policy_mutation_QA_neg.csv', "Accuracy of Policy by Iteration: Negative COT", "results/visualizations/acc_policy_neg_COT_QA.png")
    create_len_of_policy_plot('results/csv/policy_mutation_QA_neg.csv', "Length of Policy by Iteration: Negative COT", "results/visualizations/len_policy_neg_COT_QA.png")

def main():
    if sys.argv[1]=='compare':
        print('running compare')
        run_compare(sys.argv[2])
    else:
        print('running QA')
        run_QA()

if __name__ == '__main__':
    main()
