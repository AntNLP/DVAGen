import inspect
from typing import Any, Callable

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    Cache,
    GenerationMixin,
    LogitsProcessor,
    PreTrainedModel,
)
from transformers.activations import get_activation
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import LossKwargs
from typing_extensions import Unpack

from .configuration_dva import DVAConfig


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


# TODO DVAEmbeddingModel: sv embeddings (remove copy?) + phrase encoder
# class DVAEmbeddingModel(PreTrainedModel):


# extract features in phrase encoder
class DVAModel(PreTrainedModel, GenerationMixin):
    config_class = DVAConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: DVAConfig):
        super().__init__(config)
        self.config = config
        if self.config.language_model_config is not None and self.config.phrase_encoder_config is not None:
            # Initialize the DVAModel from scratch
            self.language_model = AutoModelForCausalLM.from_config(config.language_model_config)
            self.phrase_encoder = AutoModel.from_config(config.phrase_encoder_config)
            self.sv_input_embeddings = self._build_sv_embeddings(self.language_model.get_input_embeddings())
            self.sv_output_embeddings = self._build_sv_embeddings(self.language_model.get_output_embeddings())
            if self.config.use_phrase_encoder_proj:
                self.phrase_encoder_proj = self._build_phrase_encoder_proj()
        self.loss_type = "ForCausalLM"
        self.vocab_size = self.config.language_model_config.vocab_size

        self.post_init()

    @staticmethod
    def _build_sv_embeddings(src_embeddings: nn.Module) -> nn.Parameter:
        return nn.Parameter(src_embeddings.weight.clone().detach())

    def _build_phrase_encoder_proj(self) -> nn.Module:
        # TODO support more (customizable) projection layers
        return nn.Sequential(
            nn.Dropout(self.config.phrase_encoder_proj_pdrop),
            get_activation(self.config.phrase_encoder_proj_act),
            nn.Linear(self.phrase_encoder.config.hidden_size, self.language_model.config.hidden_size),
        )

    def initialize_modules(
        self, language_model_path: str, phrase_encoder_path: str, phrase_encoder_proj_path: str | None = None, **kwargs
    ):
        """Initialize the DVAModel with pre-trained language model and phrase encoder.

        If the `phrase_encoder_proj_path` is not provided, the projection layer will be initialized from scratch.
        :param language_model_path: The path to the pre-trained language model.
        :param phrase_encoder_path: The path to the pre-trained phrase encoder.
        :param phrase_encoder_proj_path: The path to the pre-trained phrase encoder projection layer.
        :param kwargs: Remaining keyword arguments.
        :return: None
        """
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_path, **kwargs)
        self.phrase_encoder = AutoModel.from_pretrained(phrase_encoder_path, **kwargs)
        self.sv_input_embeddings = self._build_sv_embeddings(self.language_model.get_input_embeddings())
        self.sv_output_embeddings = self._build_sv_embeddings(self.language_model.get_output_embeddings())
        self.config.language_model_config = self.language_model.config
        self.config.phrase_encoder_config = self.phrase_encoder.config
        if self.config.use_phrase_encoder_proj:
            self.phrase_encoder_proj = self._build_phrase_encoder_proj()
            if phrase_encoder_proj_path is not None:
                if phrase_encoder_proj_path.endswith(".safetensors"):
                    from safetensors.torch import load_file

                    state_dict = load_file(phrase_encoder_proj_path)
                else:
                    state_dict = torch.load(phrase_encoder_proj_path, map_location="cpu")

                self.phrase_encoder_proj.load_state_dict(state_dict, strict=False)

    def _get_phrase_embeddings(self, phrase_ids: torch.Tensor, phrase_attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract phrase embeddings from the phrase encoder.

        B: Batch size
        L: Sequence length
        D: Embedding dimension
        :param phrase_ids: Input ids of the phrases tokenized by the phrase tokenizer. (Shape: [B, L])
        :param phrase_attention_mask: Attention mask for the input ids. (Shape: [B, L])
        :return: Phrase embeddings of the input phrases. (Shape: [B, D])
        """
        outputs = self.phrase_encoder(input_ids=phrase_ids, attention_mask=phrase_attention_mask)
        phrase_embeddings = outputs.last_hidden_state
        phrase_end = torch.sum(phrase_attention_mask, dim=1)
        if self.config.use_phrase_encoder_proj:
            phrase_embeddings = self.phrase_encoder_proj(phrase_embeddings)

        phrase_embeddings = phrase_embeddings[range(len(phrase_embeddings)), phrase_end - 1]

        return phrase_embeddings

    @staticmethod
    def _filter_forward_params(
        forward_params: dict[str, Any], forward_func: Callable, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Update the forward method's parameters to match the target forward method's signature.

        :param forward_params: The parameters to be passed to the forward method.
        :param forward_func: The target forward function.
        :param kwargs: Additional keyword arguments to be passed to the forward method.
        """
        target_params = inspect.signature(forward_func).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in target_params.values()):
            # Check whether the language model's forward method accepts keyword arguments.
            forward_params.update(kwargs)
        else:
            # If not, filter out any unsupported parameters before calling it.
            forward_params = {k: v for k, v in forward_params.items() if k in target_params}

        return forward_params

    def get_dva_embeddings(
        self, phrase_ids: torch.Tensor | None, phrase_attention_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the input embeddings and output embeddings (LM Head) of the DVAModel.

        M: The size of the dynamic vocabulary (DV), i.e., the size of the phrase candidates.
        V: The size of the static vocabulary (SV), i.e., the vocab size of the language model.
        L: Sequence length
        :param phrase_ids: Input ids of the phrases tokenized by the phrase tokenizer. (Shape: [M, L])
        :param phrase_attention_mask: Attention mask for the input ids. (Shape: [M, L])
        :return: A tuple containing the input and output embeddings of the DVAModel. (Shape: [V+M, D])
        """
        if phrase_ids is None or phrase_attention_mask is None:
            # If the dynamic vocabulary is not provided (i.e., without phrase candidates),
            # the static vocabulary embeddings are returned as dva embeddings.
            return self.sv_input_embeddings, self.sv_output_embeddings

        dv_embeddings = [
            self._get_phrase_embeddings(
                phrase_ids[i : i + self.config.phrase_encoder_batch_size],
                phrase_attention_mask[i : i + self.config.phrase_encoder_batch_size],
            )
            for i in range(0, len(phrase_ids), self.config.phrase_encoder_batch_size)
        ]
        dva_input_embeddings = torch.cat([self.sv_input_embeddings, *dv_embeddings], dim=0)
        dva_output_embeddings = torch.cat([self.sv_output_embeddings, *dv_embeddings], dim=0)

        return dva_input_embeddings, dva_output_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        labels: torch.Tensor | None = None,
        phrase_ids: torch.Tensor | None = None,
        phrase_attention_mask: torch.Tensor | None = None,
        dva_embeds: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        """Forward function of the DVAModel.

        B: Batch size
        L: Sequence length
        P: Phrase length
        V: The size of the static vocabulary (SV), i.e., the vocab size of the language model.
        M: The size of the dynamic vocabulary (DV), i.e., the size of the phrase candidates.
        :param input_ids: Mixed input ids, including both token ids and phrase ids tokenized by DVATokenizer.
                          Indices should be in `[0, ..., V+M]` (Shape: [B, L])
        :param attention_mask: Attention mask for the input ids. (Shape: [B, L])
        :param labels: Labels for computing the causal language modeling loss.
                       Indices should either be in `[0, ..., V+M]` or `-100`. Tokens with indices set to `-100` are
                       ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., V+M]`.
        :param phrase_ids: Input ids of the phrases used to compute the dynamic vocabulary embeddings. (Shape: [M, P])
        :param phrase_attention_mask: Attention mask for the phrase ids. (Shape: [M, P])
        :param dva_embeds: Instead of passing `phrase_ids` and `phrase_attention_mask`, an alternative is to
                           pass the pre-computed input and output embeddings (SV + DV) of the DVAModel.
                           (A tuple consisting of `dva_input_embeddings` and `dva_output_embeddings`. Shape: [V+M, D])
        :return: `CausalLMOutputWithPast` containing the loss, logits, past key values, hidden states, and attentions.
        """
        if phrase_ids is not None and phrase_attention_mask is not None:
            dva_input_embeddings, dva_output_embeddings = self.get_dva_embeddings(phrase_ids, phrase_attention_mask)
        else:
            assert dva_embeds is not None, (
                "Either `phrase_ids` and `phrase_attention_mask` or `dva_embeds` must be provided to compute the "
                "DVA input and output embeddings."
            )
            dva_input_embeddings, dva_output_embeddings = dva_embeds

        if inputs_embeds is None:
            inputs_embeds = dva_input_embeddings[input_ids]

        forward_params = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "cache_position": cache_position,
        }
        forward_params = self._filter_forward_params(forward_params, self.language_model.base_model.forward, kwargs)

        outputs = self.language_model.base_model(**forward_params)
        hidden_states = outputs.last_hidden_state
        logits = hidden_states @ dva_output_embeddings.T

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=dva_output_embeddings.shape[0], **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# This class is used to mask the logits for batch inference.
class DVALogitsProcessor(LogitsProcessor):
    def __init__(self, mask_phrase_ids: list[list[int]]):
        super().__init__()
        self.mask_phrase_ids = mask_phrase_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        assert len(self.mask_phrase_ids) == scores.size(0), (
            f"Mask token ids size {len(self.mask_phrase_ids)} and scores batch size {scores.size(0)} mismatch."
        )

        for i, mask_ids in enumerate(self.mask_phrase_ids):
            if mask_ids:
                scores[i].index_fill_(0, torch.tensor(mask_ids, device=scores.device), float("-inf"))
        return scores
