import torch
from transformers import pipeline
from tqdm import tqdm
import numpy as np
import collections
import string
import re

from .metric import BaseMetric


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(a_gold, a_pred):
    """Compute F1 score between two strings."""

    def _get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_exact(a_gold, a_pred):
    """Check whether two strings are equal up to normalization."""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


class CorrectnessMetric(BaseMetric):
    def __init__(self, predictions: list[str],
                 references: list[dict],
                 model_name_path: str
    ) -> None:
        self.predictions = predictions
        self.references = references
        self.length = len(predictions)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = pipeline(
            "question-answering",
            model=model_name_path,
            device=self.device,
        )

    def compute(self) -> dict:
        em, f1, bins = [], [], []
        for idx in tqdm(range(self.length)):
            # Organize Input Data
            questions = [
                qa_pair["question"]
                for qa_pair
                in self.references[idx]["qa_pairs"]
            ]
            context = self.predictions[idx] if self.predictions[idx] else " "
            results = self.pipeline(question=questions,
                                    context=context,
                                    handle_impossible_answer=True)

            # Compute EM and F1
            loc_counter, loc_em, loc_f1 = 0, 0, 0
            for idy, res in enumerate(results):
                answers = self.references[idx]["qa_pairs"][idy]["short_answers"]
                prediction = res["answer"]

                loc_em += max([compute_exact(ans, prediction) for ans in answers])
                loc_f1 += max([compute_f1(ans, prediction) for ans in answers])
                loc_counter += 1
            em.append(loc_em / loc_counter)
            f1.append(loc_f1 / loc_counter)
            bins.append(loc_em == loc_counter)

        return {
            'QA-EM': 100 * np.mean(em),
            'QA-F1': 100 * np.mean(f1),
            # 'QA-Hit': 100 * np.mean(bins)
        }