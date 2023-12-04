from automated_llm_eval.chat_model import ChatModel, Message, Bundle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
    def __init__(self, data, task):
        """
        Initialize the AccuracyCalculator with a dictionary containing predicted and actual values.
        The dictionary should have keys 'predicted' and 'actual'.
        """
        self.data_unfiltered = data
        self.task=task
        self.data = [d for d in self.data_unfiltered if d.get('predicted') is not None]
        self.actual = [int(d.get('actual')) for d in self.data]
        # self.id = [int(d.get('id')) for d in self.data]
        # print(self.data[0])
        self.predicted = [int(d.get('predicted')) for d in self.data] #[int(d.get('predicted')) if int(d.get('actual')) != 0 else 0 for d in self.data]

    def compute_accuracy(self):
        return accuracy_score(self.actual, self.predicted)
    
    def return_incorrect(self):
        return [(d.get('id'), d.get('predicted'), d.get('actual')) for d in self.data_unfiltered if (d.get('predicted') is not None and d.get('actual') is not None and int(d.get('predicted'))!=int(d.get('actual')))]

    def compute_f1_score(self):
        return f1_score(self.actual, self.predicted, average='micro')

    def compute_precision(self):
        return precision_score(self.actual, self.predicted, average='micro')

    def compute_recall(self):
        return recall_score(self.actual, self.predicted, average='micro')

    def get_COT(self):
        """
        Compute accuracy.
        """
        correct=0
        incorrect_COT = []
        correct_COT = []
        if self.task=="compare":
            for metadata in self.data:
                human_score =int(metadata['actual'])
                agent_score = metadata['predicted']
                if not agent_score:
                    pass
                if (human_score==agent_score): #(human_score<=0 and agent_score<=0) or (human_score>=0 and agent_score>=0):
                    correct+=1
                    # correct_COT.append(metadata['statement'])
                    correct_COT.append('The agents correct reasoning for this score is as follows: '+ metadata["agent_response"])
                # elif human_score == 0:
                #     pass
                elif len(metadata["statement"])>1000:
                    statement_analysis = (
                        "A statement was summarized in the following two ways. Summary A: "
                        + metadata["human_response"]
                        + "and summary B:"
                        + metadata["llm_response"]
                        + " The summaries were compared and scored incorrectly by the agent, and the correct score should have been: "
                        + str(metadata["actual"])
                        + ". The agent's incorrect reasoning for this score is as follows: "
                        + metadata["agent_response"]
                    )
                    incorrect_COT.append(statement_analysis)
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
        elif self.task=="QA":
            for metadata in self.data:
                human_score =int(metadata['actual'])
                agent_score = metadata['predicted']
                if not agent_score:
                    pass
                if (human_score==agent_score): #(human_score<=0 and agent_score<=0) or (human_score>=0 and agent_score>=0):
                    correct+=1
                    # correct_COT.append(metadata['statement'])
                    # correct_COT.append('The agents correct reasoning for this score is as follows: '+ metadata["agent_response"])
                elif len(metadata["question"])>1000:
                    print('LONG ADJUSTMENT')
                    statement_analysis = (
                        "A question was answered in the following way: "
                        + metadata["answer"]
                        + "The appropriateness was scored incorrectly by the agent, and the correct score should have been: "
                        + str(metadata["actual"])
                        + ". The agent's incorrect reasoning for this score is as follows: "
                        + metadata["agent_response"]
                    )
                    incorrect_COT.append(statement_analysis)
                else:
                    statement_analysis = (
                        "The following question: "
                        + metadata["question"]
                        + " was answered in the following way: "
                             + metadata["answer"]
                        + "The appropriateness was scored incorrectly by the agent, and the correct score should have been: "
                        + str(metadata["actual"])
                        + ". The agent's incorrect reasoning for this score is as follows: "
                        + metadata["agent_response"]
                    )
                    incorrect_COT.append(statement_analysis)
        return incorrect_COT, correct_COT
    
    def _bootstrap_metric(self, accuracy_fn, num_samples=1000, sample_percent=0.8):
        num_examples = len(self.actual)
        metrics = []

        for _ in range(num_samples):
            sample_indices = np.random.choice(num_examples, size=int(num_examples * sample_percent), replace=True)
            sample_actual = np.take(self.actual, sample_indices)
            sample_predicted = np.take(self.predicted, sample_indices)
            metric_value = accuracy_fn(sample_actual, sample_predicted)
            metrics.append(metric_value)

        return metrics

    def compute_bootstrap_confidence_interval(self, accuracy_fn, confidence_level=0.9):

        bootstrap_metrics = self._bootstrap_metric(accuracy_fn)
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        lower_bound = np.percentile(bootstrap_metrics, lower_percentile)
        upper_bound = np.percentile(bootstrap_metrics, upper_percentile)
        return [lower_bound, upper_bound]