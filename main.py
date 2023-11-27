from automated_llm_eval.get_questions import get_questions
from automated_llm_eval.prompts import *
from private_key import *
from automated_llm_eval.test import run_test
from automated_llm_eval.model_analysis import analysis
from automated_llm_eval.visualize import *
from automated_llm_eval.policy_tuning import *
from results.html_writer import writer_html
import os

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

def run_compare(compare_type, experiment_name, reliability_type):
    if not os.path.exists('results/'+experiment_name):
        os.makedirs('results/'+experiment_name)
    else:
        pass
    policy_tuning(f"results/{experiment_name}/policy_mutation_{compare_type}.csv", compare=True, batch_size = 8, compare_type=compare_type, reliability_type =reliability_type)
    create_accuracy_plot(f"results/{experiment_name}/policy_mutation_{compare_type}.csv", "Accuracy of Policy by Iteration: Negative COT", f"results/{experiment_name}/acc_policy_neg_COT_{experiment_name}_{compare_type}.png")
    create_len_of_policy_plot(f"results/{experiment_name}/policy_mutation_{compare_type}.csv", "Length of Policy by Iteration: Negative COT", f"results/{experiment_name}/len_policy_neg_COT_{experiment_name}_{compare_type}.png")
    writer_html(f"results/{experiment_name}/policy_mutation_{compare_type}.csv", f"results/{experiment_name}/policy_mutation_{compare_type}.html")
def run_QA():
    policy_tuning('results/csv/policy_mutation_QA_neg.csv', compare=False, batch_size = 1, compare_type = 'pls')
    create_accuracy_plot('results/csv/policy_mutation_QA_neg.csv', "Accuracy of Policy by Iteration: Negative COT", "results/visualizations/acc_policy_neg_COT_QA.png")
    create_len_of_policy_plot('results/csv/policy_mutation_QA_neg.csv', "Length of Policy by Iteration: Negative COT", "results/visualizations/len_policy_neg_COT_QA.png")

def main():
    datasets = ['iii', 'chq', 'pls']
    if sys.argv[1]=='compare':
        run_compare(sys.argv[2], "pos_COT_overfitting_on_training_metric_high", sys.argv[3])
    else:
        print('running QA')
        run_QA()

if __name__ == '__main__':
    main()
