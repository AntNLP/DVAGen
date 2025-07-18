import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metric import BaseMetric


class PerplexityMetric(BaseMetric):
    def __init__(
        self,
        predictions: list[str],
        model_name_or_path: str = None,
        batch_size: int = 16,
        add_start_token: bool = True,
        max_length: int = None,
    ):
        self.predictions = predictions
        self.model_name_or_path = model_name_or_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.batch_size = batch_size
        self.add_start_token = add_start_token
        self.max_length = max_length
        assert self.model_name_or_path is not None, "Model must be provided for the perplexity metric."

    def compute(self) -> dict[str, float]:
        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if self.tokenizer.pad_token is None and self.batch_size > 1:
            existing_special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert len(existing_special_tokens) > 0, (
                "If batch_size > 1, model must have at least one special token to use for padding. Please use a "
                "different model or set batch_size=1."
            )
            # assign one of the special tokens to also be the pad token
            self.tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if self.add_start_token and self.max_length:
            # leave room for <BOS> token to be added:
            assert self.tokenizer.bos_token is not None, (
                "Input model must already have a BOS token if using add_start_token=True. Please use a different "
                "model, or set add_start_token=False"
            )
            max_tokenized_len = self.max_length - 1
        else:
            max_tokenized_len = self.max_length

        encodings = self.tokenizer(
            self.predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if self.add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(torch.ge(attn_masks.sum(1), 2)), (
                "When add_start_token=False, each input text must be at least two tokens long. Run with "
                "add_start_token=True if inputting strings of only one token, and remove all empty input strings."
            )

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in tqdm(range(0, len(encoded_texts), self.batch_size)):
            end_index = min(start_index + self.batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if self.add_start_token:
                bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(
                    self.device
                )
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexity": np.mean(ppls)}
