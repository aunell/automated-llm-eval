from langchain.prompts import PromptTemplate

# Not all tasks are Q&A, for example compare A vs. B.
# so we need to have a more generic agent prompt template where we can
# put any kind of task we want into it.

# Solution: 2-stage template construction; Fill out task template first, then insert into {task} in agent template
TASK_TMPL = """\
Source: {source}
Summary A: {summary_a}
Summary B: {summary_b}"""


def agent_predict_prompt_template(
    instruction: str = "{instruction}", task: str = "{task}"
) -> str:
    return f"""\
You are an expert AI agent that strictly follows the instructions given below to complete the task.

INSTRUCTION:
{instruction}

TASK:
{task}

ANSWER:
Score: """


AGENT_PREDICT_SUMMARY_TMPL = agent_predict_prompt_template(task=TASK_TMPL)
AGENT_PREDICT_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["instruction", "source", "summary_a", "summary_b"],
    template=AGENT_PREDICT_SUMMARY_TMPL,
)

example_prompt = AGENT_PREDICT_SUMMARY_PROMPT.format(
    instruction="insert the policy rules here",
    source="this is the source text from which both summaries are generated",
    summary_a="summary aaa",
    summary_b="summary bbb",
)

TASK_WITH_CORRECT_ANSWER_TMPL = """\
Source: {source}
Summary A: {summary_a}
Summary B: {summary_b}
Answer: {answer}"""

TASK_WITH_INCORRECT_ANSWER_TMPL = """\
Source: {source}
Summary A: {summary_a}
Summary B: {summary_b}
Answer: {answer}
Correct Answer: {target}"""

POLICY_MUTATE_PROMPT_TMPL = """\
You are an expert AI agent that is able to adapt policies to ensure optimal conformity from large language models.
A language model was given an original policy and asked to label the statements accordingly.
--------------------------
ORIGINAL POLICY:
{original_policy}
--------------------------
Below is a list of examples that have been assigned the correct label.
CORRECT ANSWERS:
{correct_answers}
--------------------------
Below is a list of examples that have been assigned the incorrect label. The correct label is provided.  The reason for why each example is incorrect is also provided. 
INCORRECT ANSWERS:
{incorrect_answers}
--------------------------
Given this information, adjust the original policy so that the model will maintain its accuracy on the correct labelled examples but change its response on the incorrectly labeled examples so that it will label these examples correctly.
Ensure that any information added to the policy is novel and the policy remains concise, complete, and correct.

REVISED POLICY:
"""

POLICY_MUTATE_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "original_policy",
        "correct_answers",
        "incorrect_answers",
    ],
    template=POLICY_MUTATE_PROMPT_TMPL,
)
# original_policy = policy text used to produce correct/incorrect answers
# correct_answers = a list of "TASK_WITH_CORRECT_ANSWER_TMPL" that is joined into a string
#                   and delimited with "\n\n". We need to generate this list
#                   from the LLM-predictor's answers.
# incorrect_answers = a list of "TASK_WITH_INCORRECT_ANSWER_TMPL" that is joined into a string
#                   and delimited with "\n\n". We need to generate this list
#                   from the LLM-predictor's answers and also provide the correct answer.
