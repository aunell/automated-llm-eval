from automated_llm_eval.get_data_helping_functions import get_mode_score_compare
from automated_llm_eval.chat_model import Message

from automated_llm_eval.prompts import (
    COMPARE_AGENT_PROMPT,
    GPT_SYSTEM_PROMPT,
    POLICY_MUTATE_PROMPT_TEMPLATE,
    QA_AGENT_PROMPT,
    SCORE_RETRIEVAL_PROMPT,
    prompt_improvement_character_prompt,
    score_retrieval_character_prompt,
)
def construct_message(example: dict, current_policy: str, task: str):
    match task:
        case "compare":
            return construct_compare_message(example, current_policy)
        case "QA":
            return construct_QA_message(example, current_policy)
        case _:
            return None

def construct_compare_message(example: dict, current_policy: str) -> Message:
    idx_to_mode = get_mode_score_compare()
    human_label = idx_to_mode[int(example["idx"])]
    statement = example["inputs"]
    human_response = example["target"]
    llm_response = example["output"]

    compare_gpt_prompt = COMPARE_AGENT_PROMPT.format(
        source=statement,
        current_policy=current_policy,
        summary_a=human_response,
        summary_b=llm_response,
    )
    system_message = GPT_SYSTEM_PROMPT
    user_message = compare_gpt_prompt
    message = Message(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        metadata={
            "actual": human_label,
            "statement": statement,
            "human_response": human_response,
            "llm_response": llm_response,
            "id": example["id"]
        },
    )
    return message

def construct_QA_message(example: dict, current_policy: str) -> Message:
    human_label = example["Label"]
    question = example["Prompt text"]
    answer = example["Response"]

    QA_gpt_prompt = QA_AGENT_PROMPT.format(
        question=question,
        answer=answer,
        current_policy=current_policy
    )
    system_message = GPT_SYSTEM_PROMPT
    user_message = QA_gpt_prompt
    message = Message(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        metadata={
            "actual": human_label,
            "question": question,
            "answer": answer,
            "id": example["id"]
        },
    )
    return message

def construct_label_extraction_message(explanation: str) -> Message:
    "Extract Agent Label from free-text Explanations."
    system_message = score_retrieval_character_prompt
    user_message = SCORE_RETRIEVAL_PROMPT.format(response=explanation)
    return Message(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        metadata={}
    )