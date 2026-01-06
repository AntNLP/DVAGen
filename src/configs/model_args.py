from dataclasses import dataclass
from enum import Enum, auto

from src.models.phrase import FMMConfig, NTokensConfig, NWordsConfig

from ..models.configuration_dva import DVAConfig


@dataclass
class DVAModelArguments(DVAConfig.to_dataclass()):
    model_name_or_path: str | None = None
    language_model_path: str | None = None
    phrase_encoder_path: str | None = None


@dataclass
class FMMConfig:
    phrase_gap: int = 1
    phrase_kept_ratio: float = 0.1
    add_phrase_len: int = 10


@dataclass
class NWordsConfig:
    phrase_interval_begin: int = 0
    phrase_interval_end: int = 10000
    add_phrase_len: int = 10
    add_black: int = 10


@dataclass
class NTokensConfig:
    phrase_interval_begin: int = 0
    phrase_interval_end: int = 10000
    add_phrase_len: int = 10


@dataclass
class BaseModelArguments:
    model_name_or_path: str
    phrase_encoder_name_or_path: str
    backbone_name_or_path: str
    backbone_config_file: str


class PhraseSamplerType(Enum):
    fmm = auto()  # forward maximum matching
    n_words = auto()
    n_tokens = auto()


@dataclass
class PhraseSamplerArguments:
    phrase_sampler_type: str
    fmm_config: FMMConfig | None = None
    n_words_config: NWordsConfig | None = None
    n_tokens_config: NTokensConfig | None = None

    def __post_init__(self):
        pass
        # if self.phrase_sampler_type == PhraseSamplerType.fmm:
        #     if self.fmm_config is None:
        #         raise ValueError("fmm_config must be provided")
        # elif self.phrase_sampler_type == PhraseSamplerType.n_words:
        #     if self.n_words_config is None:
        #         raise ValueError("n_words_kwargs must be provided")
        # elif self.phrase_sampler_type == PhraseSamplerType.n_tokens:
        #     if self.n_tokens_config is None:
        #         raise ValueError("n_tokens_config must be provided")
        # else:
        #     raise ValueError(f"Invalid phrase_sampler_type: {self.phrase_sampler_type}")
