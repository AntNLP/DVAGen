from .metric import BaseMetric


class BytesPerTokenMetric(BaseMetric):
    def __init__(
        self,
        predictions: list[str],
        predictions_ids: list[list[int | list[int]]],
    ):
        self.predictions = predictions
        self.predictions_ids = predictions_ids
        assert len(self.predictions) == len(self.predictions_ids), "Predictions and ids must have the same length."

    def compute(self) -> dict[str, float]:
        total_ids = 0
        total_bytes = 0
        for pred, pred_ids in zip(self.predictions, self.predictions_ids):
            total_ids += len(pred_ids)
            total_bytes += len(pred.encode("utf-8"))

        bpt = total_bytes / total_ids if total_ids else 0.0
        return {"BytesPerToken": bpt}
