from langchain.prompts.prompt import PromptTemplate

GPT_SYSTEM_PROMPT = (
    "You are an expert AI agent, possessing an in-depth knowledge and expertise within "
    "the healthcare and medical domain."
)

# The first prompt template is a simple QA template, which is used when querying the model for the first time
DEFAULT_QA_PROMPT_TMPL = (
    "You are an expert AI agent that strictly follows the guidelines given below.\n"
    "Answer the following questions: \n"
    "Original Question: {question}\n"
    "Answer: "
)

DEFAULT_QA_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=DEFAULT_QA_PROMPT_TMPL,
)

# This is the refine template. Meaning, when the model is queried multiple times, and it has previous answers as well as the responses
# from various agents, it can use it to refine its answers
DEFAULT_REFINE_PROMPT_TMPL = (
    "You are an expert AI agent that strictly follows the guidelines given below.\n"
    "----------------------------------------------------\n"
    "Description of SafetyGPT, EthicsGPT, and ClinicianGPT agents:\n"
    "- SafetyGPT is an expert AI agent with in-depth knowledge and expertise in healthcare and medical safety.\n"
    "- EthicsGPT is a knowledgeable AI assistant specializing in healthcare and medical ethics.\n"
    "- ClinicianGPT is a proficient AI clinician, possessing the knowledge and experience of a seasoned medical doctor.\n"
    "----------------------------------------------------\n"
    "Original Question: {question}\n"
    "Existing Answer: {existing_answer}\n"
    "----------------------------------------------------\n"
    "Given the original question and existing answer, here are the responses from SafetyGPT, EthicsGPT, and ClinicianGPT agents:\n"
    "- SafetyGPT Response: {safety_gpt_response}\n"
    "- EthicsGPT Response: {ethics_gpt_response}\n"
    "- ClinicianGPT Response: {clinician_gpt_response}\n"
    "----------------------------------------------------\n"
    "Now, based on the provided context, please refine the Existing Answer to ensure it satisfies all three agents (SafetyGPT, EthicsGPT, ClinicianGPT).\n"
    "These agents are considered satisfied only when all their responses are positive and there are no remaining issues flagged by them.\n"
    "Please keep refining the answer until all agents are satisfied with a score of 8 or above. Refinement of the response should ONLY increase the numerical score of SafetyGPT, EthicsGPT, and ClinicianGPT or keep it the same."
    "If one or more of the agents already present a score of 8 or above, focus more on the suggestions provided by the remaining agents that are not satisfied."
    "If all agents are satisfied and have a score of 8 or higher, then do not refine the answer and just return 'DONE!' exactly and nothing else.\n"
    "If any agent is not satisfied, continue refining the answer accordingly.\n"
    "Finally, please ensure that the refined answer is not incomplete and allow for the entire answer to finish generating."
    "Refined Answer: "
)

DEFAULT_REFINE_PROMPT = PromptTemplate(
    input_variables=[
        "question",
        "existing_answer",
        "safety_gpt_response",
        "ethics_gpt_response",
        "clinician_gpt_response",
    ],
    template=DEFAULT_REFINE_PROMPT_TMPL,
)


# This is the prompt used by the multiple agents (ethics, safety, and clinician)
DEFAULT_AGENT_PROMPT_TMPL = (
    "You are an expert AI agent that strictly follows the guidelines given below.\n"
    "--------------------------\n"
    "Original Question: {question}\n"
    "Existing Answer: {existing_answer}\n"
    "--------------------------\n"
    "Your Previous Response: {agent_response}\n"
    "--------------------------\n"
    "Guidelines to Follow:\n"
    "{agent_guideline}\n"
    "--------------------------\n"
    "If the existing answer appropriately addresses the question, aligns with your guidelines, and addresses concerns expressed in your previous response, respond 'Existing Answer meets the guideline and the score is' followed by the associated score."
    "If it doesn't, respond 'The score is' followed by the associated score and clearly specify the aspect of the answer that falls short and provide specific suggestions for improvement. Please ensure that suggested refinements to the existing answer make the new answer have a score that is greater than or equal to the existing answer's score.\n"
)

DEFAULT_AGENT_PROMPT = PromptTemplate(
    input_variables=[
        "question",
        "existing_answer",
        "agent_response",
        "agent_guideline",
    ],
    template=DEFAULT_AGENT_PROMPT_TMPL,
)

SCORE_RETRIEVAL_PROMPT = PromptTemplate(
    input_variables=["response"],
    template=(
        "Read the following response and respond with just the given score number. You should only return a numerical value with no words, letters, or punctuation:{response}."
    ),
)

score_retrieval_character_prompt = "You are an AI system that can read text and return the numerical score that is included in the text."

prompt_improvement_character_prompt = "You are an AI system that is an expert at prompt engineering and can improve a prompt for large language models so that the model is better able to return the expected results."


# POLICY_MUTATE_PROMPT_TMPL = """\
# You are an expert AI agent that is able to adapt policies to ensure optimal conformity from large language models.
# A language model was given an original policy and asked to label the statements accordingly.
# --------------------------
# ORIGINAL POLICY:
# {original_policy}
# --------------------------
# Below are explanations by the model for scoring responses where the laguage model scored the response correctly:
# CORRECT EXPLANATION:
# {correct_answers}
# --------------------------
# Below is a list of examples that have been assigned the incorrect label. The correct label is provided.  The reason for why each example is incorrect is also provided. 
# INCORRECT ANSWERS:
# {incorrect_answers}
# --------------------------
# Given this information, adjust the original policy so that the model will maintain its accuracy on the correct labelled examples but change its response on the incorrectly labeled examples so that it will label these examples correctly.
# Ensure that any information added to the policy is novel and the policy remains concise, complete, and correct. You may use specific examples. Focus on changing the policy so that the labels incorrectly generated by the model are corrected.
# Aim to improve the accuracy of the language model.

# REVISED POLICY:
# """
POLICY_MUTATE_PROMPT_TMPL = """\
You are an expert AI agent that is able to adapt policies to ensure optimal conformity from large language models.
A language model was given an original policy and asked to label the statements accordingly.
--------------------------
ORIGINAL POLICY:
{original_policy}
--------------------------
Below is a list of examples that have been assigned the incorrect label. The correct label is provided.  The reason for why each example is incorrect is also provided. 
INCORRECT ANSWERS:
{incorrect_answers}
--------------------------
Given this information, adjust the original policy so that the model will maintain its accuracy on the correct labelled examples but change its response on the incorrectly labeled examples so that it will label these examples correctly.
Ensure that any information added to the policy is novel and the policy remains concise, complete, and correct. You may use specific examples. Focus on changing the policy so that the labels incorrectly generated by the model are corrected.
Aim to improve the accuracy of the language model.

REVISED POLICY:
"""

POLICY_MUTATE_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "original_policy",
        # "correct_answers",
        "incorrect_answers",
    ],
    template=POLICY_MUTATE_PROMPT_TMPL,
)

COMPARE_AGENT_PROMPT_TMPL = (
    "Source Text: {source}\n"
    "Summary A: {summary_a}\n"
    "Summary B: {summary_b}\n"
    "Follow this policy to produce a score comparing summary A and summary B: {current_policy}"
)

COMPARE_AGENT_PROMPT = PromptTemplate(
    input_variables=["source", "current_policy", "summary_a", "summary_b"],
    template=COMPARE_AGENT_PROMPT_TMPL,
)

QA_AGENT_PROMPT_TMPL = (
    "Question: {question}\n"
    "Answer: {answer}\n"
    "Follow this policy to produce a score for the above statement: {current_policy}"
)

QA_AGENT_PROMPT = PromptTemplate(
    input_variables=["question", "answer", "current_policy"],
    template=QA_AGENT_PROMPT_TMPL,
)

GENERATE_REAS0NING_QA = """\
You are an expert AI agent that is able to explain why statements are {label}. Please explain why the following answer is an {label} response to the given question.
Question: {question}
Answer: {answer}
"""

GENERATE_REAS0NING_QA_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
                "question",
         "answer",
         "label"
    ],
    template=GENERATE_REAS0NING_QA,
)

MUTATION_REAS0NING_QA = """\
You are an expert AI agent that is able to adapt policies to ensure optimal conformity from large language models.
Please adjust the following policy to account for the following reasoning. Policy: {original_policy}
--------------------------
The current policy is flawed and labels the following question-answer pair incorrectly. Question: {question}. Answer: {answer}
--------------------------
Please use this reasoning to adjust the original policy. Reasoning: {reasoning}

REVISED POLICY:
"""

MUTATION_REAS0NING_QA_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "original_policy",
        "question",
        "answer",
        "reasoning"
    ],
    template=MUTATION_REAS0NING_QA,
)
