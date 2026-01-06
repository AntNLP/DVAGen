import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


# fix seed
random.seed(906)


@dataclass
class Document:
    content: str = None
    token_ids: list[int] = None
    id: int = None


@dataclass
class Phrase:
    content: str
    is_phrase: bool
    src_doc_id: int | None = None  # ID of the document this phrase comes from


# TODO: add base phrase type config


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


class BasePhraseSampler(ABC):
    @abstractmethod
    def __init__(self): ...

    @abstractmethod
    def sample(self, document: Document) -> list[Phrase]: ...


class NTokenPhraseSampler(BasePhraseSampler):
    # TODO: replace kwargs with specific config classes
    def __init__(self, config, **kwargs):
        self.config = config
        assert self.config is not None, "config must be provided"
        self.tokenizer = kwargs.get("tokenizer", None)
        self.random_up = kwargs.get("random_up", 12)
        self.random_low = kwargs.get("random_low", 8)
        self.phrase_max_length = kwargs.get("phrase_max_length", 5)
        self.phrase_num = 0

    def sample(self, document: Document) -> list[Phrase]:
        phrases = []
        tokens = self.tokenizer.tokenize(document.content)
        end_index = len(tokens)
        now = 0
        while now < end_index:
            start = random.randint(now + self.random_low, now + self.random_up)
            end = start + random.randint(2, self.phrase_max_length)
            if end > end_index or start > end:
                content = self.tokenizer.convert_tokens_to_string(tokens[now:end_index])
                phrases.append(Phrase(content=content, is_phrase=False))
                break
            content = self.tokenizer.convert_tokens_to_string(tokens[now:start])
            phrases.append(Phrase(content=content, is_phrase=False))

            # content = self.tokenizer.lm_tokenizer.convert_tokens_to_string(tokens[now:start])
            # phrases.append(Phrase(content=content,is_phrase=False))
            # if end > end_index or start > end:
            #     break

            phrases.append(
                Phrase(
                    content=self.tokenizer.convert_tokens_to_string(tokens[start:end]),
                    is_phrase=True,
                )
            )
            now = end
        return phrases

    def sample_negative(self, document: Document) -> list[Phrase]: ...
