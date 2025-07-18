import numpy as np
import spacy
from tqdm import tqdm

from .metric import BaseMetric


class RepMetric(BaseMetric):
    DEFAULT_NGRAM_SIZES = [2, 3, 4]

    def __init__(self, predictions: list[str]):
        self.predictions = predictions
        self.spacy_model = spacy.load("en_core_web_sm")

    @staticmethod
    def get_rep_keyname(ngram_size: int) -> str:
        return f"Rep-{ngram_size}"

    def calculate_ngram_repetition(self, sentence: str, ngram_size: int = 1) -> float:
        tokens = [token.text for token in self.spacy_model(sentence)]

        ngrams = []
        for i in range(len(tokens) - ngram_size + 1):
            ngrams.append(tuple(tokens[i : i + ngram_size]))

        if not ngrams:
            return 1.0

        unique_ngrams = set(ngrams)
        repetition_rate = 1 - len(unique_ngrams) / len(ngrams)

        return repetition_rate

    def compute(self) -> dict[str, float]:
        repetition_scores_per_text = []

        for prediction in tqdm(self.predictions):
            text_scores = {}
            for ngram_size in self.DEFAULT_NGRAM_SIZES:
                score_key = self.get_rep_keyname(ngram_size)
                text_scores[score_key] = self.calculate_ngram_repetition(prediction, ngram_size)
            repetition_scores_per_text.append(text_scores)
        overall_scores = {}
        overall_diversity = 1.0

        for ngram_size in self.DEFAULT_NGRAM_SIZES:
            score_key = self.get_rep_keyname(ngram_size)

            avg_repetition = np.mean([scores[score_key] for scores in repetition_scores_per_text])

            diversity_for_ngram = 1 - avg_repetition
            overall_diversity *= diversity_for_ngram

            overall_scores[score_key] = avg_repetition
        overall_scores["Diversity"] = overall_diversity

        return overall_scores
