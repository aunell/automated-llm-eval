import pandas as pd
from automated_llm_eval.general_helping_functions import compare_responses

def writer_html(csv_file, output):
    diff_tables = []
    df = pd.read_csv(csv_file)
    responses = df.iloc[0].to_list()

    for i in range(len(responses)-2):
        result = compare_responses(responses[i], responses[i+1])
        diff_tables.append(result)
        diff_tables.append("<hr>")
    combined_html = "".join(diff_tables)
    with open(f"{output}.html", "w") as html_file:
        html_file.write(combined_html)