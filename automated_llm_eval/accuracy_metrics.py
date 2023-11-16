from automated_llm_eval.chat_model import ChatModel, Message, Bundle
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

from automated_llm_eval.prompts import (
    COMPARE_AGENT_PROMPT,
    GPT_SYSTEM_PROMPT,
    POLICY_MUTATE_PROMPT_TEMPLATE,
    QA_AGENT_PROMPT,
    SCORE_RETRIEVAL_PROMPT,
    prompt_improvement_character_prompt,
    score_retrieval_character_prompt,
)
class AccuracyMetrics:
    def __init__(self, data):
        """
        Initialize the AccuracyCalculator with a dictionary containing predicted and actual values.
        The dictionary should have keys 'predicted' and 'actual'.
        """
        self.data = data
        self.actual = [d.get('actual') for d in self.data]
        self.predicted = [d.get('predicted') for d in self.data]

##SKLEARN NOT WORKING
    # def compute_accuracy(self):
    #     return accuracy_score(self.actual, self.predicted)

    # def compute_f1_score(self):
    #     return f1_score(self.actual, self.predicted)

    # def compute_precision(self):
    #     return precision_score(self.actual, self.predicted)

    # def compute_recall(self):
    #     return recall_score(self.actual, self.predicted)
    
    def compute_accuracy(self):
        correct_predictions = sum(1 for a, p in zip(self.actual, self.predicted) if a == p)
        total_predictions = len(self.actual)
        return correct_predictions / total_predictions

    def compute_f1_score(self):
        true_positives = sum(1 for a, p in zip(self.actual, self.predicted) if a == p == 1)
        false_positives = sum(1 for a, p in zip(self.actual, self.predicted) if a == 0 and p == 1)
        false_negatives = sum(1 for a, p in zip(self.actual, self.predicted) if a == 1 and p == 0)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1

    def compute_precision(self):
        true_positives = sum(1 for a, p in zip(self.actual, self.predicted) if a == p == 1)
        false_positives = sum(1 for a, p in zip(self.actual, self.predicted) if a == 0 and p == 1)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        return precision

    def compute_recall(self):
        true_positives = sum(1 for a, p in zip(self.actual, self.predicted) if a == p == 1)
        false_negatives = sum(1 for a, p in zip(self.actual, self.predicted) if a == 1 and p == 0)

        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        return recall

    def get_COT(self):
        """
        Compute accuracy.
        """
        correct=0
        incorrect_COT = []
        correct_COT = []
        for metadata in self.data:
            human_score =metadata['actual']
            agent_score = metadata['predicted']
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
                    + str(metadata["actual"])
                    + ". The agent's incorrect reasoning for this score is as follows: "
                    + metadata["agent_response"]
                )
                incorrect_COT.append(statement_analysis)
        return incorrect_COT, correct_COT
    
    def _bootstrap_metric(self, num_samples=1000, sample_percent=0.8):
        num_examples = len(self.actual)
        metrics = []

        def accuracy_bootstrap(actual, predicted):
            correct_predictions = sum(1 for a, p in zip(actual, predicted) if a == p)
            total_predictions = len(actual)
            return correct_predictions / total_predictions

        for _ in range(num_samples):
            sample_indices = np.random.choice(num_examples, size=int(num_examples * sample_percent), replace=True)
            sample_actual = np.take(self.actual, sample_indices)
            sample_predicted = np.take(self.predicted, sample_indices)
            metric_value = accuracy_bootstrap(sample_actual, sample_predicted)
            metrics.append(metric_value)

        return metrics

    def compute_bootstrap_confidence_interval(self, confidence_level=0.9):
        bootstrap_metrics = self._bootstrap_metric()
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        lower_bound = np.percentile(bootstrap_metrics, lower_percentile)
        upper_bound = np.percentile(bootstrap_metrics, upper_percentile)
        return [lower_bound, upper_bound]

# def get_score(agent_response: str):
#     SCORE_RETRIEVAL = SCORE_RETRIEVAL_PROMPT.format(response=agent_response)
#     model_score = ChatModel(
#             model="gpt-3.5-turbo-1106", temperature=0.1, top_p=0.5, max_tokens=700, seed=42
#         )
#     response_score_string = model_score.create_chat_completion(
#             score_retrieval_character_prompt, SCORE_RETRIEVAL, output_format="simple"
#         )
#     try:
#         response_int = int(response_score_string)
#         return response_int
#     except:
#         return None