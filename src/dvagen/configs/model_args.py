from dataclasses import dataclass
from enum import Enum

from ..models.configuration_dva import DVAConfig


class PhraseSamplerType(Enum):
    FMM = "FMM"
    N_WORDS = "N_WORDS"
    N_TOKENS = "N_TOKENS"


@dataclass
class SamplerConfig:
    phrase_sampler_type: PhraseSamplerType = PhraseSamplerType.N_TOKENS
    # n_tokens
    sampler_model_path: str | None = None

    # n_token and n_words
    sampler_random_up: int | None = None
    sampler_random_low: int | None = None
    phrase_max_length: int | None = None

    # fmm
    fmm_embedding_model_path: str | None = None
    fmm_data_file: str | None = None
    fmm_vector_store_path: str | None = None
    fmm_save_vector_store_path: str | None = None
    fmm_min_length: int = 2
    fmm_max_length: int = 16


@dataclass
class DVAModelArguments(DVAConfig.to_dataclass(), SamplerConfig):
    model_name_or_path: str | None = None
    language_model_path: str | None = None
    phrase_encoder_path: str | None = None
