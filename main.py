import pandas as pd
from tqdm import tqdm
from langchain.prompts.prompt import PromptTemplate
import re
import numpy as np
import matplotlib.pyplot as plt

from get_questions import get_questions
from create_chat_completion import create_chat_completion
from prompts import *
from model_performance import model_performance
from private_key import *
from test import run_test
from model_analysis import analysis
from visualize import *

questions =get_questions()
# run_number = 3

openai_token = key["open-ai"]

engine_options = ["gpt-3.5-turbo", "gpt-4"]
judge_options = ["gpt-3.5-turbo", "gpt-4"]

run_test(engine_options, judge_options)

for engine in engine_options:
    for engine_judge in judge_options:
        analysis(engine, engine_judge, engine+ ' + ' +engine_judge)

iteration_number = []
number_answered= []
names = []
error_answered= []
error_iteration = []
for engine in engine_options:
    for engine_judge in judge_options:
        df = analysis(engine, engine_judge, run_number, engine+ ' + ' +engine_judge)
        df.columns= df.columns.get_level_values(1)
        # make_spider_plot(df, engine, engine_judge)
        iteration_number.append(df.at['Iterations', 'Mean'])
        number_answered.append(df.at['Number Answered', 'Mean'])
        error_answered.append(np.std(df.at['Number Answered', 'Samples']))
        error_iteration.append(np.std(df.at['Iterations', 'Samples']))
        names.append(engine+'+'+engine_judge)

make_spider_plot(engine_options, judge_options)

create_bar_plots(iteration_number, names, error_iteration, 'iteration_number')
create_bar_plots(number_answered, names, error_answered, 'number_answered') 