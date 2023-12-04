import logging
import random
import pandas as pd
from sklearn.metrics import accuracy_score

from automated_llm_eval.chat_model import ChatModel, Message
from automated_llm_eval.general_helping_functions import (
    save_dict_as_csv,
    compute_metrics,
    compare_responses,
    find_average,
    editDistance,
    select_batch
)
from automated_llm_eval.get_data_helping_functions import get_data_split, get_policy_file
from automated_llm_eval.message_helping_functions import construct_message, construct_label_extraction_message

from automated_llm_eval.prompts import (
    POLICY_MUTATE_PROMPT_TEMPLATE,
    SCORE_RETRIEVAL_PROMPT,
    prompt_improvement_character_prompt
)
from automated_llm_eval.utils import sidethread_event_loop_async_runner
from automated_llm_eval.accuracy_metrics import AccuracyMetrics


def generate_for_dataset(
    dataset: dict,
    current_policy: str,
    batch_size: int,
    task: str,
    model: str = "gpt-3.5-turbo-1106",
    temperature: float = 0,
    top_p: float = 0.5,
    max_tokens: int = 700,
    seed: int = 42,
    num_concurrent: int = 20,
) -> list[Message]:
    """Selects batch, formats messages, asynchronously makes LLM calls,
    extract label from agent rationale."""
    batch = select_batch(dataset=dataset, batch_size=batch_size, seed=seed)
    msg_list = []
    for example in batch:
        msg = construct_message(example=example, current_policy=current_policy, task = task)
        msg_list += [msg]
    
    # Create ChatModel
    model = ChatModel(
        model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed
    )
    result_bundles = sidethread_event_loop_async_runner(
        async_function=model.async_chat_completions(
            messages_list=msg_list, num_concurrent=num_concurrent, output_format="bundle"
        )
    )
    agent_responses = [bundle.response_message for bundle in result_bundles if bundle is not None]
    agent_label_extraction_messages = [
        construct_label_extraction_message(agent_response) for agent_response in agent_responses
    ]
    agent_label_results = sidethread_event_loop_async_runner(
        async_function=model.async_chat_completions(
            messages_list=agent_label_extraction_messages,
            num_concurrent=num_concurrent,
            output_format="simple",
        )
    )

    def check_agent_label(label_str: str) -> int | None:
        try:
            response_int = int(label_str)
            return response_int
        except Exception:
            return None

    agent_labels = [check_agent_label(agent_label) for agent_label in agent_label_results]

    updated_msg_list = []
    for example_msg, agent_response, agent_label, bundle in zip(
        msg_list, agent_responses, agent_labels, result_bundles
    ):
        updated_msg = example_msg.metadata | {
            "agent_response": agent_response,
            "predicted": agent_label,
            "bundle": bundle
        }
        updated_msg_list += [updated_msg]
    return updated_msg_list


def check_policy_accuracy(dataset, current_policy, batch_size, task, seed):
    """
    return numerical accuracy score as well as COT statements
    score, incorrect statements, correct statements
    """
    results = generate_for_dataset(
        dataset, current_policy=current_policy, batch_size=batch_size, seed=seed, task=task
    )
    accuracy_object = AccuracyMetrics(results, task)
    accuracy_dictionary = compute_metrics(accuracy_object)
    incorrect_classified_tuple = accuracy_object.return_incorrect()
    print('NUMBER WRONG', len(incorrect_classified_tuple))
    confidence_interval = accuracy_object.compute_bootstrap_confidence_interval(accuracy_score)
    return accuracy_dictionary["accuracy"], accuracy_dictionary["COT"][0], accuracy_dictionary["COT"][1], confidence_interval, incorrect_classified_tuple


def policy_tuning(output: str, task: str, batch_size: int, compare_type, reliability_type):
    # add_fleiss_column("scored_examples/vanDeenCollapsed.csv")
    logging.basicConfig(level=logging.INFO, filename=f'{output}.log', filemode='w')
    score = 0.0
    train_data, test_data = get_data_split(task, compare_type, reliability_type)
    current_policy = get_policy_file(task)
    # score_before, _, _ , confidence_interval_before= check_policy_accuracy(test_data, current_policy, batch_size=1, seed=42, task="compare")
    # print("test score before", score_before, "confidence before", confidence_interval_before)
    data = {}
    i = 0
    responses = []
    diff_tables = []
    convergence = False
    score_list = []
    epsilon = .03
    while (score < 1 and i < 5 and not convergence):
        print("score is", score, "and iteration is:", i)
        _, incorrect_labelled, correct_labelled , _, _ = check_policy_accuracy(train_data, current_policy, batch_size, seed=0, task=task)
        score, _,_ , val_confidence_interval, incorrect_classified_tuple = check_policy_accuracy(test_data, current_policy, batch_size=1, seed=0, task=task)
        if len(score_list)==3:
            if abs(find_average(score_list)-score)<epsilon:
                convergence=True
            else:
                score_list.pop(0)
                score_list.append(score)
        AGENT_IMPROVEMENT = POLICY_MUTATE_PROMPT_TEMPLATE.format(
            original_policy=current_policy,
            # correct_answers=correct_labelled,
            incorrect_answers=incorrect_labelled,
        )
        model = ChatModel(
            model="gpt-3.5-turbo-1106", temperature=0.1, top_p=0.5, max_tokens=700, seed=42
        )
        if i==0:
            distance=0
        else:
            distance = editDistance(current_policy, data[i-1]["current_policy"])
        data[i] = {"current_policy": current_policy, "score": score, "lower_limit": val_confidence_interval[0], "upper_limit" :val_confidence_interval[1], "distance": distance, "missed statements": incorrect_classified_tuple}

        try:
            current_policyNew = model.create_chat_completion(
                prompt_improvement_character_prompt, AGENT_IMPROVEMENT, output_format="simple"
            )
            if current_policyNew is not None:
                responses.append((current_policy, current_policyNew))
                current_policy = current_policyNew
        except Exception as e:
            print(e)
        save_dict_as_csv(data, output)
        i += 1

    for previous_response, response in responses:
        result = compare_responses(previous_response, response)
        diff_tables.append(result)
        diff_tables.append("<hr>")
    combined_html = "".join(diff_tables)
    with open(f"{output}.html", "w") as html_file:
        html_file.write(combined_html)

    # score_after, incorrect_labelled, correct_labelled , confidence_interval_after = check_policy_accuracy(test_data, current_policy, batch_size=1, seed=42, task="compare")
    # data["final scores"] = [score_before, score_after, confidence_interval_before, confidence_interval_after]
    save_dict_as_csv(data, output)
    return current_policy
