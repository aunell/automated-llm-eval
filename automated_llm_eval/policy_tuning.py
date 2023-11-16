import random

from automated_llm_eval.bundle_accuracy import BundleAccuracy
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


def create_agent_response_message(
    current_policy, source_text, compare, statement_a=None, statement_b=None
):
    gpt_system_prompt = GPT_SYSTEM_PROMPT
    # model = ChatModel(
    #     model="gpt-3.5-turbo-1106", temperature=0.5, top_p=0.5, max_tokens=700, seed=42
    # )
    if compare:
        compare_gpt_prompt = COMPARE_AGENT_PROMPT.format(
            source=source_text,
            current_policy=current_policy,
            summary_a=statement_a,
            summary_b=statement_b,
        )
        system_message = gpt_system_prompt
        user_message = compare_gpt_prompt
        message = Message(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        )
        return message
        # gpt_response = model.async_chat_completion()
        # gpt_response = model.create_chat_completion(
        #     gpt_system_prompt, compare_gpt_prompt, output_format="simple"
        # )
    else:
        safety_gpt_prompt = QA_AGENT_PROMPT.format(
            statement=source_text, current_policy=current_policy
        )
        system_message = gpt_system_prompt
        user_message = safety_gpt_prompt
        message = Message(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        )
        return message
        # gpt_response = model.create_chat_completion(
        #     gpt_system_prompt, safety_gpt_prompt, output_format="simple"
        # )
    # return gpt_response


def select_batch(dataset: dict, batch_size: int, compare: bool, seed: int = 42) -> list:
    examples = list(dataset.values())
    random.Random(seed).shuffle(examples)
    batch = examples[: len(examples) // batch_size]
    return batch


def construct_compare_message(example: dict, current_policy: str) -> Message:
    gpt_system_prompt = GPT_SYSTEM_PROMPT

    idx_to_mode = get_mode_score_compare()
    human_score = idx_to_mode[int(example["idx"])]
    statement = example["inputs"]
    human_response = example["target"]
    llm_response = example["output"]

    compare_gpt_prompt = COMPARE_AGENT_PROMPT.format(
        source=statement,
        current_policy=current_policy,
        summary_a=human_response,
        summary_b=llm_response,
    )
    system_message = gpt_system_prompt
    user_message = compare_gpt_prompt
    message = Message(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        metadata={
            "human_score": human_score,
            "statement": statement,
            "human_response": human_response,
            "llm_response": llm_response,
            # TODO: add agent score after API call
        },
    )
    return message


def construct_message(example: dict, current_policy: str, compare: bool):
    if compare:
        return construct_compare_message(example, current_policy)
    else:
        # TODO: safety message construction
        return None


# def construct_message(example: dict, current_policy: str, compare: bool):
#     # Form Agent Response Score & Metadata
#     if compare:
#         idx_to_mode = get_mode_score_compare()
#         statement = example["inputs"]
#         human_response = example["target"]
#         agent_response = example["output"]

#         agent_response_score = create_agent_response(
#             current_policy, statement, compare, human_response, agent_response
#         )
#         metadata = {
#             "human_score": idx_to_mode[int(example["idx"])],
#             "statement": statement,
#             "human_response": human_response,
#             "agent_response": agent_response,
#             "agent_response_score": agent_response_score,
#         }

#     else:
#         agent_response_score = create_agent_response(current_policy, statement, compare)
#         metadata = {
#             "human_score": int(example["Human Label (Dev)"]),
#             "statement": example["LLM-Generated Statements"],
#             "agent_response_score": agent_response_score,
#         }

#     # Create Messages
#     system_message = score_retrieval_character_prompt
#     user_message = SCORE_RETRIEVAL_PROMPT.format(response=agent_response_score)
#     messages = [
#         {"role": "system", "content": system_message},
#         {"role": "user", "content": user_message},
#     ]

#     return Message(messages=messages, metadata=metadata)


def make_message_to_extract_score_from_explanation(explanation: str):
    # Extract Agent Label from free-text Explanations
    system_message = score_retrieval_character_prompt
    user_message = SCORE_RETRIEVAL_PROMPT.format(response=explanation)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    return Message(messages=messages)


def generate_for_dataset(
    dataset: dict,
    current_policy: str,
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
        msg = construct_message(example=example, current_policy=current_policy, compare=compare)
        msg_list += [msg]

    # Create ChatModel
    model = ChatModel(
        model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed
    )

    # Generate Agent Label + Explanation on Task for every example in batch
    result = sidethread_event_loop_async_runner(
        async_function=model.async_chat_completions(
            messages_list=msg_list, num_concurrent=num_concurrent, output_format="bundle"
        )
    )
    agent_responses = [x.response_message for x in result]
    # Extract agent label from responses
    agent_label_extraction_messages = [
        make_message_to_extract_score_from_explanation(x) for x in agent_responses
    ]
    result = sidethread_event_loop_async_runner(
        async_function=model.async_chat_completions(
            messages_list=agent_label_extraction_messages,
            num_concurrent=num_concurrent,
            output_format="simple",
        )
    )

    # Check that the label is an integer
    def check_agent_label(label_str: str) -> int | None:
        try:
            response_int = int(label_str)
            return response_int
        except Exception:
            return None

    agent_labels = [check_agent_label(x) for x in result]

    # metadata = [x.metadata for x in result]
    updated_msg_list = []
    for example_msg, agent_response, agent_label in zip(msg_list, agent_responses, agent_labels):
        updated_msg = example_msg.metadata | {
            "agent_response": agent_response,
            "agent_label": agent_label,
        }
        updated_msg_list += [updated_msg]

    return updated_msg_list


def check_policy_accuracy(dataset, current_policy, batchsize, compare):
    """
    return numerical accuracy score as well as COT statements
    score, incorrect statements, correct statements
    """
    print("generating results")
    results = generate_for_dataset(
        dataset, current_policy=current_policy, batch_size=4, compare=compare
    )
    print("results generated")
    return BundleAccuracy(results).accuracy()
    # for example in results
    #     print("WE ARE ON THE ", k, "th example")
    #     if compare:
    #         human_score = idx_to_mode[int(example["idx"])]
    #         statement = example["inputs"]
    #         agent_response = example["output"]
    #         human_response = example["target"]
    #         agent_response_score = create_agent_response(
    #             current_policy, statement, compare, human_response, agent_response
    #         )
    #     else:
    #         statement = example["LLM-Generated Statements"]
    #         human_score = int(example["Human Label (Dev)"])
    #         agent_response_score = create_agent_response(current_policy, statement, compare)

    #     SCORE_RETRIEVAL = SCORE_RETRIEVAL_PROMPT.format(response=agent_response_score)
    #     model_score = ChatModel(
    #         model="gpt-3.5-turbo-1106", temperature=0.1, top_p=0.5, max_tokens=700, seed=42
    #     )
    #     response_score_string = model_score.create_chat_completion(
    #         score_retrieval_character_prompt, SCORE_RETRIEVAL, output_format="simple"
    #     )
    #     try:
    #         response_score = int(response_score_string)
    #         if response_score != human_score:
    #             print("incorrect result from agent")
    #             statement_analysis = (
    #                 "The following statement: "
    #                 + statement
    #                 + " was given a score of: "
    #                 + str(response_score)
    #                 + " by the agent, but the correct score should have been: "
    #                 + str(human_score)
    #                 + ". The agent's reasoning for this score is as follows: "
    #                 + agent_response_score
    #             )
    #             incorrect_labelled.append(statement_analysis)
    #         else:
    #             correct_labelled.append(statement)
    #     except Exception:
    #         print("RESPONSE SCORE STRING ERROR:", response_score_string)
    #     k += 1
    # score = len(correct_labelled) / (len(correct_labelled) + len(incorrect_labelled)) or 0
    # return current_policy, score, correct_labelled, incorrect_labelled


def policy_tuning(output, compare, batch_size, compare_type="iii"):
    score = 0.0
    train_data, test_data = get_data_split(compare, compare_type)
    current_policy = get_policy_file(compare)
    score_before, _, _ = check_policy_accuracy(test_data, current_policy, batch_size, compare)
    print("test score before", score_before)
    data = {}
    i = 0

    while score < 0.9 and i < 10:
        print("score is", score, "and iteration is:", i)
        score, incorrect_labelled, correct_labelled = check_policy_accuracy(
            train_data, current_policy, batch_size, compare
        )
        data[i] = [current_policy, score]
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

    score_after, _, _ = check_policy_accuracy(test_data, current_policy, batch_size, compare)
    data["final scores"] = [score_before, score_after]
    save_as_csv(data, output)
    return current_policy
