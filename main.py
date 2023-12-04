from automated_llm_eval.prompts import *
from private_key import *
from automated_llm_eval.visualize import *
from automated_llm_eval.policy_tuning import *
from results.html_writer import writer_html
import os

import sys
   
openai_token = key["open-ai"]

def run_experiment(task, experiment_name, reliability_type, compare_type=None):
    if not os.path.exists('results/'+experiment_name):
        os.makedirs('results/'+experiment_name)
    else:
        pass
    policy_tuning(f"results/{experiment_name}/policy_mutation_{compare_type}.csv", task, batch_size = 5, compare_type=compare_type, reliability_type =reliability_type)
    create_accuracy_plot(f"results/{experiment_name}/policy_mutation_{compare_type}.csv", "Accuracy of Policy by Iteration: Negative COT", f"results/{experiment_name}/acc_policy_neg_COT_{experiment_name}_{compare_type}.png")
    create_len_of_policy_plot(f"results/{experiment_name}/policy_mutation_{compare_type}.csv", "Length of Policy by Iteration: Negative COT", f"results/{experiment_name}/len_policy_neg_COT_{experiment_name}_{compare_type}.png")
    writer_html(f"results/{experiment_name}/policy_mutation_{compare_type}.csv", f"results/{experiment_name}/policy_mutation_{compare_type}.html")
    visualize_overlap(f"results/{experiment_name}/policy_mutation_{compare_type}.csv", f"results/{experiment_name}/overlap_{compare_type}.png")

def main():
    run_experiment(task=sys.argv[1], experiment_name="check_code_test", reliability_type = sys.argv[3] if len(sys.argv) > 3 else None, compare_type= sys.argv[2] if len(sys.argv) > 2 else None)

if __name__ == '__main__':
    main()
