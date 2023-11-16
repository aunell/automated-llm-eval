from automated_llm_eval.chat_model import ChatModel, Message, Bundle
from automated_llm_eval.prompts import (
    COMPARE_AGENT_PROMPT,
    GPT_SYSTEM_PROMPT,
    POLICY_MUTATE_PROMPT_TEMPLATE,
    QA_AGENT_PROMPT,
    SCORE_RETRIEVAL_PROMPT,
    prompt_improvement_character_prompt,
    score_retrieval_character_prompt,
)
class BundleAccuracy:
    def __init__(self, data):
        """
        Initialize the AccuracyCalculator with a dictionary containing predicted and actual values.
        The dictionary should have keys 'predicted' and 'actual'.
        """
        self.data = data

    def accuracy(self):
        """
        Compute accuracy.
        """
        correct=0
        incorrect_COT = []
        correct_COT = []
        for metadata in self.data:
            human_score =metadata['human_label']
            agent_score = metadata['agent_label']
            if not agent_score:
                pass
            if int(human_score)==agent_score:
                correct+=1
                correct_COT.append(metadata['statement'])
            else:
                statement_analysis = (
                    "The following statement: "
                    + metadata["statement"]
                    + " was summarized in the following two ways. Summary A: "
                    + metadata["human_response"]
                    + "and summary B:"
                    + metadata["llm_response"]
                    + " The summaries were compared and scored incorrectly by the agent, and the correct score should have been: "
                    + str(metadata["human_label"])
                    + ". The agent's incorrect reasoning for this score is as follows: "
                    + metadata["agent_response"]
                )
                incorrect_COT.append(statement_analysis)
        return correct/(len(incorrect_COT)+correct) or 0, incorrect_COT, correct_COT
    


def get_score(agent_response: str):
    SCORE_RETRIEVAL = SCORE_RETRIEVAL_PROMPT.format(response=agent_response)
    model_score = ChatModel(
            model="gpt-3.5-turbo-1106", temperature=0.1, top_p=0.5, max_tokens=700, seed=42
        )
    response_score_string = model_score.create_chat_completion(
            score_retrieval_character_prompt, SCORE_RETRIEVAL, output_format="simple"
        )
    try:
        response_int = int(response_score_string)
        return response_int
    except:
        return None