import logging
import random
from sklearn.metrics import accuracy_score

from automated_llm_eval.chat_model import ChatModel, Message
from automated_llm_eval.policy_helping_functions import (
    get_data_split,
    get_mode_score_compare,
    get_policy_file,
    save_as_csv,
    confidence_interval,
    compute_metrics
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
from automated_llm_eval.accuracy_metrics import AccuracyMetrics
logger = logging.getLogger("PolicyTuneLogger")


def select_batch(dataset: dict, batch_size: int, seed: int = 42) -> list:
    examples = list(dataset.values())
    random.Random(seed).shuffle(examples)
    batch = examples[: len(examples) // batch_size]
    return batch


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
        },
    )
    return message


def construct_safety_message(example: dict, current_policy: str) -> Message:
    statement = example["inputs"]
    safety_gpt_prompt = QA_AGENT_PROMPT.format(statement=statement, current_policy=current_policy)
    system_message = GPT_SYSTEM_PROMPT
    user_message = safety_gpt_prompt
    message = Message(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    )
    return message


def construct_message(example: dict, current_policy: str, task: str):
    match task:
        case "compare":
            return construct_compare_message(example, current_policy)
        case "safety":
            return construct_safety_message(example, current_policy)
        case _:
            return None


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
    logger.info("Selecting Batch...")
    batch = select_batch(dataset=dataset, batch_size=batch_size, seed=seed)

    logger.info("Create Message Prompts + Metadata")
    msg_list = []
    for example in batch:
        msg = construct_message(example=example, current_policy=current_policy, task = task)
        msg_list += [msg]

    # Create ChatModel
    model = ChatModel(
        model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed
    )

    logger.info("Generate Agent Response (Label + CoT Explanation) for every example in batch")
    result_bundles = sidethread_event_loop_async_runner(
        async_function=model.async_chat_completions(
            messages_list=msg_list, num_concurrent=num_concurrent, output_format="bundle"
        )
    )
    agent_responses = [bundle.response_message for bundle in result_bundles]

    logger.info("Use LLM to Extract agent label from responses")
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

    logger.info("Check that the label is an integer")

    def check_agent_label(label_str: str) -> int | None:
        try:
            response_int = int(label_str)
            return response_int
        except Exception:
            return None

    agent_labels = [check_agent_label(agent_label) for agent_label in agent_label_results]

    logger.info("Update Messages metadata with the Generated Agent Response + Extracted Label")
    updated_msg_list = []
    for example_msg, agent_response, agent_label, bundle in zip(
        msg_list, agent_responses, agent_labels, result_bundles
    ):
        updated_msg = example_msg.metadata | {
            "agent_response": agent_response,
            "predicted": agent_label,
            "bundle": bundle,
        }
        updated_msg_list += [updated_msg]
    # Return all original messages with uppdated metadata
    return updated_msg_list


def check_policy_accuracy(dataset, current_policy, batchsize, task, seed):
    """
    return numerical accuracy score as well as COT statements
    score, incorrect statements, correct statements
    """
    logging.info("generating results")
    results = generate_for_dataset(
        dataset, current_policy=current_policy, batch_size=batchsize, seed=seed, task=task
    )
    logging.info("results generated")
    # print(results)
    accuracy_object = AccuracyMetrics(results)
    accuracy_dictionary = compute_metrics(accuracy_object)
    confidence_interval = accuracy_object.compute_bootstrap_confidence_interval(accuracy_score)
    return accuracy_dictionary["accuracy"], accuracy_dictionary["COT"][0], accuracy_dictionary["COT"][1], confidence_interval


def policy_tuning(output, compare, batch_size, compare_type="iii"):
    score = 0.0
    train_data, test_data = get_data_split(compare, compare_type)
    current_policy = get_policy_file(compare)
    score_before, _, _ , confidence_interval_before= check_policy_accuracy(test_data, current_policy, batch_size, seed=42, task="compare")
    print("test score before", score_before, "confidence before", confidence_interval_before)
    data = {}
    i = 0

    while score < 0.9 and i < 10:
        print("score is", score, "and iteration is:", i)
        _, incorrect_labelled, correct_labelled , _ = check_policy_accuracy(train_data, current_policy, batch_size, seed=i, task="compare")
        val_score, _,_ , val_confidence_interval = check_policy_accuracy(test_data, current_policy, batch_size, seed=0, task="compare")

        # current_policy, score, lower_bound, upper_bound, correct_labelled, incorrect_labelled = find_score_and_confidence_interval(train_data, current_policy, batch_size, seed=i, task="compare")
        data[i] = [current_policy, val_score, val_confidence_interval[0], val_confidence_interval[1]]
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

    score_after, incorrect_labelled, correct_labelled , confidence_interval_after = check_policy_accuracy(test_data, current_policy, batch_size, seed=42, task="compare")
    data["final scores"] = [score_before, score_after, confidence_interval_before, confidence_interval_after]
    save_as_csv(data, output)
    return current_policy
