import random

from automated_llm_eval.chat_model import ChatModel, Message
from automated_llm_eval.policy_helping_functions import (
    get_data_split,
    get_mode_score_compare,
    get_policy_file,
    save_as_csv,
)
from automated_llm_eval.prompts import (
    COMPARE_AGENT_PROMPT,
    GPT_SYSTEM_PROMPT,
    POLICY_MUTATE_PROMPT_TEMPLATE,
    QA_AGENT_PROMPT,
    SCORE_RETRIEVAL_PROMPT,
    prompt_improvement_character_prompt,
    score_retrieval_character_prompt,
)
from automated_llm_eval.utils import sidethread_event_loop_async_runner


def create_agent_response(current_policy, source_text, compare, statement_a=None, statement_b=None):
    gpt_system_prompt = GPT_SYSTEM_PROMPT
    model = ChatModel(
        model="gpt-3.5-turbo-1106", temperature=0.5, top_p=0.5, max_tokens=700, seed=42
    )
    if compare:
        compare_gpt_prompt = COMPARE_AGENT_PROMPT.format(
            source=source_text,
            current_policy=current_policy,
            summary_a=statement_a,
            summary_b=statement_b,
        )
        gpt_response = model.create_chat_completion(
            gpt_system_prompt, compare_gpt_prompt, output_format="simple"
        )
    else:
        safety_gpt_prompt = QA_AGENT_PROMPT.format(
            statement=source_text, current_policy=current_policy
        )
        gpt_response = model.create_chat_completion(
            gpt_system_prompt, safety_gpt_prompt, output_format="simple"
        )
    return gpt_response


def select_batch(dataset: dict, batch_size: int, seed: int = 42) -> list:
    examples = list(dataset.values())
    examples = random.Random(seed).shuffle(examples)
    batch = examples[: len(examples) // batch_size]
    return batch


def construct_message(example: dict, current_policy: str, compare: bool):
    # Form Agent Response Score & Metadata
    if compare:
        idx_to_mode = get_mode_score_compare()
        statement = example["inputs"]
        human_response = example["target"]
        agent_response = example["output"]
        agent_response_score = create_agent_response(
            current_policy, statement, compare, human_response, agent_response
        )
        metadata = {
            "human_score": idx_to_mode[int(example["idx"])],
            "statement": statement,
            "human_response": human_response,
            "agent_response": agent_response,
            "agent_response_score": agent_response_score,
        }

    else:
        agent_response_score = create_agent_response(current_policy, statement, compare)
        metadata = {
            "human_score": int(example["Human Label (Dev)"]),
            "statement": example["LLM-Generated Statements"],
            "agent_response_score": agent_response_score,
        }

    # Create Messages
    system_message = score_retrieval_character_prompt
    user_message = SCORE_RETRIEVAL_PROMPT.format(response=agent_response_score)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    return Message(messages=messages, metadata=metadata)


def generate_for_dataset(
    dataset: dict,
    batch_size: int,
    compare: bool,
    model: str = "gpt-3.5-turbo-1106",
    temperature: float = 0.1,
    top_p: float = 0.5,
    max_tokens: int = 700,
    seed: int = 42,
    num_concurrent: int = 5,
    output_format: str = "bundle",
):
    "Selects batch, formats messages, asynchronously makes LLM calls"
    # Select Batch
    batch = select_batch(dataset=dataset, batch_size=batch_size, compare=compare)
    # Create Message Prompts + Metadata
    msg_list = []
    for example in batch:
        msg = construct_message(example=example, compare=compare)
        msg_list += [msg]
    # Create ChatModel
    model = ChatModel(
        model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed
    )
    # Async ChatCompletion on another thread
    result = sidethread_event_loop_async_runner(
        async_function=model.async_chat_completions(
            messages_list=msg_list, num_concurrent=num_concurrent, output_format=output_format
        )
    )
    return result


def check_policy_accuracy(dataset, current_policy, batchsize, compare):
    incorrect_labelled = []
    correct_labelled = []
    if compare:
        idx_to_mode = get_mode_score_compare()
    k = 0
    shuffled_data = list(dataset.values())
    random.shuffle(shuffled_data)
    for example in shuffled_data[: len(shuffled_data) // batchsize]:
        print("WE ARE ON THE ", k, "th example")
        if compare:
            human_score = idx_to_mode[int(example["idx"])]
            statement = example["inputs"]
            agent_response = example["output"]
            human_response = example["target"]
            agent_response_score = create_agent_response(
                current_policy, statement, compare, human_response, agent_response
            )
        else:
            statement = example["LLM-Generated Statements"]
            human_score = int(example["Human Label (Dev)"])
            agent_response_score = create_agent_response(current_policy, statement, compare)

        SCORE_RETRIEVAL = SCORE_RETRIEVAL_PROMPT.format(response=agent_response_score)
        model_score = ChatModel(
            model="gpt-3.5-turbo-1106", temperature=0.1, top_p=0.5, max_tokens=700, seed=42
        )
        response_score_string = model_score.create_chat_completion(
            score_retrieval_character_prompt, SCORE_RETRIEVAL, output_format="simple"
        )
        try:
            response_score = int(response_score_string)
            if response_score != human_score:
                print("incorrect result from agent")
                statement_analysis = (
                    "The following statement: "
                    + statement
                    + " was given a score of: "
                    + str(response_score)
                    + " by the agent, but the correct score should have been: "
                    + str(human_score)
                    + ". The agent's reasoning for this score is as follows: "
                    + agent_response_score
                )
                incorrect_labelled.append(statement_analysis)
            else:
                correct_labelled.append(statement)
        except Exception:
            print("RESPONSE SCORE STRING ERROR:", response_score_string)
        k += 1
    score = len(correct_labelled) / (len(correct_labelled) + len(incorrect_labelled)) or 0
    return current_policy, score, correct_labelled, incorrect_labelled


def policy_tuning(output, compare, batch_size, compare_type="iii"):
    score = 0.0
    train_data, test_data = get_data_split(compare, compare_type)
    current_policy = get_policy_file(compare)
    _, score_before, _, _ = check_policy_accuracy(test_data, current_policy, batch_size, compare)
    data = {}
    i = 0

    while score < 0.9 and i < 10:
        print("score is", score)
        current_policy, score, correct_labelled, incorrect_labelled = check_policy_accuracy(
            train_data, current_policy, batch_size, compare
        )
        data[i] = [current_policy, score, correct_labelled, incorrect_labelled]
        AGENT_IMPROVEMENT = POLICY_MUTATE_PROMPT_TEMPLATE.format(
            original_policy=current_policy,
            correct_answers=correct_labelled,
            incorrect_answers=incorrect_labelled,
        )
        model = ChatModel(
            model="gpt-3.5-turbo-1106", temperature=0.5, top_p=0.5, max_tokens=700, seed=42
        )
        try:
            current_policyNew = model.create_chat_completion(
                prompt_improvement_character_prompt, AGENT_IMPROVEMENT, output_format="simple"
            )
            if current_policyNew is not None:
                current_policy = current_policyNew
        except Exception:
            pass
        save_as_csv(
            data, f"results/csv/policy_mutation_snapshot_{compare_type}_compare{compare}.csv"
        )
        i += 1

    (
        _,
        score_after,
        _,
        _,
    ) = check_policy_accuracy(test_data, current_policy, batch_size, compare)
    data["final scores"] = [score_before, score_after]
    save_as_csv(data, output)
    return current_policy
