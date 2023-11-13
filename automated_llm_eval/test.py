
import os
from automated_llm_eval.model_performance import *
from private_key import *

openai_token = key["open-ai"]
run_number = 3

def run_test(engine_options, judge_options):
    for engine in engine_options:
        for engine_judge in judge_options:
            base_directory = "./data"+f"/{engine} + {engine_judge}"
            os.makedirs(base_directory, exist_ok=True)
            for i in range(run_number):
                file_name =base_directory+f"/{engine}_{engine_judge}_Model_{i}.csv"
                model_performance(engine, engine_judge, openai_token, file_name)