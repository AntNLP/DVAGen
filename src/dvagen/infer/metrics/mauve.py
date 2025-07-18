import torch
from mauve import compute_mauve

from .metric import BaseMetric


class MauveMetric(BaseMetric):
    def __init__(
        self,
        predictions: list[str],
        references: list[str] = None,
        model_name_or_path: str = None,
        batch_size: int = 16,
    ):
        self.predictions = predictions
        self.references = references
        self.model_name_or_path = model_name_or_path
        self.device_id = 0 if torch.cuda.is_available() else -1
        self.batch_size = batch_size
        assert self.references is not None, "References must be provided for the MAUVE metric."
        assert len(self.predictions) == len(self.references), "Predictions and references must have the same length."
        assert self.model_name_or_path is not None, "Model must be provided for featuring."

    def compute(self) -> dict[str, float]:
        mauve_score = compute_mauve(
            p_text=self.predictions,
            q_text=self.references,
            device_id=self.device_id,
            featurize_model_name=self.model_name_or_path,
            mauve_scaling_factor=2.0,
            verbose=True,
            batch_size=self.batch_size,
        )

        return {"MAUVE": mauve_score.mauve}
