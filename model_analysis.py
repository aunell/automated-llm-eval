import statistics as stats
import pandas as pd
from test import run_number

# run_number = 3

def model_runs_list_creation(engine, engine_judge):
    model_runs_list=[]
    for k in range(run_number):
        model_runs_list.append(pd.read_csv(f"/work/data3/{engine} + {engine_judge}/{engine}_{engine_judge}_Model_{k}.csv"))
    return model_runs_list

def analysis(engine, engine_judge, model_name='Model'):
    # I wrote these super quickly --> In theory this can be better written (will fix later)
    model_runs_list = model_runs_list_creation(engine, engine_judge)

    # Number of Iterations
    model_iterations = []
    for run in model_runs_list:
        iter_number = len(list(run["Iteration #"]))
        model_iterations.append(iter_number)

    iter_mean = round(stats.mean(model_iterations),3)
    iter_stdev = round(stats.stdev(model_iterations),3)


    # Average Agent Scores
    agents = ["SafetyGPT Score", "EthicsGPT Score", "ClinicianGPT Score"]
    agent_model_avg_scores = {}
    for agent in agents:
        model_avg_scores = []
        for run in model_runs_list:
            agent_scores = list(run[agent])
            avg_score = round(stats.mean(agent_scores),3)
            model_avg_scores.append(avg_score)
    
        agent_model_avg_scores[agent] = [model_avg_scores, round(stats.mean(model_avg_scores),3), round(stats.stdev(model_avg_scores),3)]


    # Number of Qs Answered
    numb_answered = []
    for run in model_runs_list:
        run_dup = run.copy()
        run_dup.loc[len(run_dup.index)] = [0,0,0,0,0,0,0,0,0,0]
        Questions = list(run_dup["Question"])
        answered = 0
        for i in range(len(run)):
            curr_question = Questions[i]
            next_question = Questions[i+1]
            SafetyGPT_score = list(run_dup["SafetyGPT Score"])[i]
            EthicsGPT_score = list(run_dup["EthicsGPT Score"])[i]
            ClinicianGPT_score = list(run_dup["ClinicianGPT Score"])[i]

            if curr_question != next_question:
                if SafetyGPT_score >= 8:
                    if EthicsGPT_score >= 8:
                        if ClinicianGPT_score >= 8:
                            answered += 1
        
        numb_answered.append(answered)

    numb_answered_mean = round(stats.mean(numb_answered),3)
    numb_answered_stdev = round(stats.stdev(numb_answered),3)



    final_dict = {
    'Mean':[iter_mean,agent_model_avg_scores["SafetyGPT Score"][1], agent_model_avg_scores["EthicsGPT Score"][1], agent_model_avg_scores["ClinicianGPT Score"][1],numb_answered_mean],
    'StDev':[iter_stdev,agent_model_avg_scores["SafetyGPT Score"][2], agent_model_avg_scores["EthicsGPT Score"][2], agent_model_avg_scores["ClinicianGPT Score"][2],numb_answered_stdev],
    'Samples': [model_iterations, agent_model_avg_scores["SafetyGPT Score"][0], agent_model_avg_scores["EthicsGPT Score"][0], agent_model_avg_scores["ClinicianGPT Score"][0], numb_answered]
}
    analysis_results = pd.DataFrame(final_dict)

    analysis_results.columns = [[model_name,model_name,model_name],['Mean','StDev', 'Samples']]
    analysis_results.index = ["Iterations", "Avg Safety Score", "Avg Ethics Score", "Avg Clinician Score", "Number Answered"]

    # display(analysis_results)

    return(analysis_results)