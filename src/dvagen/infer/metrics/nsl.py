from transformers import AutoTokenizer

from .metric import BaseMetric


class NSLMetric(BaseMetric):
    def __init__(
        self, predictions: list[str], predictions_ids: list[list[int | list[int]]], model_name_or_path: str = None
    ):
        self.predictions = predictions
        self.predictions_ids = predictions_ids
        self.model_name_or_path = model_name_or_path
        assert len(self.predictions) == len(self.predictions_ids), "Predictions and ids must have the same length."
        assert self.model_name_or_path is not None, "Model must be provided for the NSL metric."
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def compute(self) -> dict[str, float]:
        total_ids = 0
        total_tokens = 0
        for pred, pred_ids in zip(self.predictions, self.predictions_ids):
            total_ids += len(pred_ids)
            total_tokens += len(self.tokenizer.tokenize(pred))

        nsl = total_ids / total_tokens if total_tokens else 0.0
        return {"NSL": nsl}
