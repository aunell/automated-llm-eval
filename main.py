from get_questions import get_questions
from prompts import *
from private_key import *
from test import run_test
from model_analysis import analysis
from visualize import *
from policy_tuning import *
   
openai_token = key["open-ai"]

def general_response_experiment():
    questions =get_questions()

    engine_options = ["gpt-3.5-turbo", "gpt-4"]
    judge_options = ["gpt-3.5-turbo", "gpt-4"]

    run_test(engine_options, judge_options)

    for engine in engine_options:
        for engine_judge in judge_options:
            analysis(engine, engine_judge, engine+ ' + ' +engine_judge)

    create_plots(engine_options, judge_options)

def policy_tuning_experiment(agent, openai_token):
    policy_tuning(agent, openai_token)

policy_tuning("gpt-4", openai_token)