from langchain.prompts.prompt import PromptTemplate

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
    input_variables=["question", 
                     "existing_answer", 
                     "safety_gpt_response", 
                     "ethics_gpt_response", 
                     "clinician_gpt_response"
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
    # "In light of your previous response and the provided agent guidelines, please assess the existing answer critically. Please rank the answer on a scale of 1 to 10, with 1 being an exceedingly poor answer and 10 being an answer that meets all the guidelines."
    "If the existing answer appropriately addresses the question, aligns with your guidelines, and addresses concerns expressed in your previous response, respond 'Existing Answer meets the guideline and the score is' followed by the associated score."
    "If it doesn't, respond 'The score is' followed by the associated score and clearly specify the aspect of the answer that falls short and provide specific suggestions for improvement. Please ensure that suggested refinements to the existing answer make the new answer have a score that is greater than or equal to the existing answer's score.\n"
)

DEFAULT_AGENT_PROMPT = PromptTemplate(
    input_variables=["question", 
                     "existing_answer", 
                     "agent_response", 
                     "agent_guideline", 
                    ],
    template=DEFAULT_AGENT_PROMPT_TMPL,
)

SCORE_RETRIEVAL_PROMPT = PromptTemplate(
                input_variables=["response" ],
                template=("Read the following response and respond with just the given score number. You should only return a numerical value with no words, letters, or punctuation:{response}."))

score_retrieval_character_prompt = "You are an AI system that can read text and return the numerical score that is included in the text."

prompt_improvement_character_prompt = "You are an AI system that is an expert at prompt engineering and can improve a prompt for large language models so that the model is better able to return the expected results."


ITERATIVE_AGENT_IMPROVEMENT_PROMPT_TMPL = (
    "You are an expert AI agent that is able to adapt policies to ensure optimal conformity from large language models."
    "A language model was given an original policy and asked to score statements accordingly. Below is the original policy and the list of correctly scored statements.\n"
    "--------------------------\n"
    "Original Policy: {original_policy}\n"
    "--------------------------\n"
    "Correct Answers: {correct_answers}\n"
    "--------------------------\n"
    # "For the correctly scored answers, below are statements that were correctly labelled along with the reasoning for why the agent labelled the response correctly."
    # "Correct Answers: {correct_answers}\n"
    "--------------------------\n"
    "For the incorrectly scored answers, below are statements that were incorrectly labelled, as well as what their correct label should be. The reasoning for why the agent labelled the response incorrectly is also provided."
    "Incorrect Answers: {incorrect_answers}\n"
    "--------------------------\n"
    "Given this information, adjust the original policy so that the model will maintain its accuracy on the correct labelled answers but change its calculated score on the incorrect answers to the provided correct label."
    "Ensure that any information added to the policy is novel and the policy remains concise, complete, and correct."
)

ITERATIVE_AGENT_IMPROVEMENT_PROMPT = PromptTemplate(
    input_variables=["correct_answers", 
                     "incorrect_answers", 
                     "original_policy",
                    ],
    template=ITERATIVE_AGENT_IMPROVEMENT_PROMPT_TMPL,
)