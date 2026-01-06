import torch
from torch import tensor
from transformers import AutoTokenizer

from src.configs.model_args import PhraseSamplerArguments
from src.models.phrase import Document, NTokenPhraseSampler, Phrase


# TODO:
# 1. 预测时的phrase如果训练没见过怎么办
# class DVATokenizer(PreTrainedTokenizer):
#     def __init__(self, vocab_file: str, **kwargs):
#         super().__init__(vocab_file, **kwargs)
class DVATokenizer:
    def __init__(self, phrase_sampler_config: PhraseSamplerArguments, static_vocab: int, **kwargs):
        self.lm_tokenizer = AutoTokenizer.from_pretrained(
            kwargs.get("model_name_or_path", "gpt2"),
        )
        self.static_vocab = static_vocab
        self.phrase_tokenizer = AutoTokenizer.from_pretrained(
            kwargs.get("phrase_encoder_name_or_path", "gpt2"),
        )
        self.phrase_sampler_config = phrase_sampler_config
        assert self.phrase_sampler_config is not None, "phrase_sampler_config must be provided"

        if self.lm_tokenizer.pad_token_id is None:
            self.lm_tokenizer.pad_token_id = self.lm_tokenizer.eos_token_id
        if self.phrase_tokenizer.pad_token_id is None:
            self.phrase_tokenizer.pad_token_id = self.phrase_tokenizer.eos_token_id

        # TODO: no if else
        if self.phrase_sampler_config.phrase_sampler_type == "n_tokens":
            self.sampler = NTokenPhraseSampler(tokenizer=self.lm_tokenizer, config=self.phrase_sampler_config)
        elif self.phrase_sampler_config.phrase_sampler_type == "n_words":
            pass
        elif self.phrase_sampler_config.phrase_sampler_type == "fmm":
            pass

    # TODO: (str, token or pharse)
    def tokenize(self, text: str):
        if self.phrase_sampler_config.phrase_sampler_type == "n_tokens":
            # TODO: optimize config class
            doc = Document(content=text)
            phrases = self.sampler.sample(doc)
            return phrases
        elif self.phrase_sampler_config.phrase_sampler_type == "n_words":
            pass
        elif self.phrase_sampler_config.phrase_sampler_type == "fmm":
            pass
        return super().tokenize(text)

    def encode(self, phrases: list[Phrase], **kwargs):
        input_ids = []
        phrase_ids = []  # list[list[int]]
        for phrase in phrases:
            if phrase.is_phrase:
                input_ids.append(self.static_vocab + len(phrase_ids))  # Unique ID for phrase
                phrase_ids.append(self.phrase_tokenizer.encode(phrase.content, add_special_tokens=False))
            else:
                input_ids.extend(self.lm_tokenizer.encode(phrase.content, add_special_tokens=False))
        return {
            "input_ids": input_ids,
            "phrase_ids": phrase_ids,
        }

    def batch_encode(self, phrases_list: list[list[Phrase]], logits_mask: bool, **kwargs):
        """
        batch
        input [0, 1, 1002, 1] len(phrase[0])=3 [0, 1, *2*]
        input [0, 1, 1001, 1, 1, 1] len(phrase[1])=4 [0, *1*, 2, 3]

        after collate
        input [
            [0, 1, 1002, 1],
            [0, 1, 1001+cumsum(len(phrase[:1])), 1, 1, 1]
        ]
        len(total_phrase)=7

        [0,1,2,3,<pad>,<pad>]               [1,1,1,1,0,0]
        [2,3,454,52,63,744]                       [1,1,1,1,1,1]
        [0,<pad>,<pad>,<pad>,<pad>,<pad>]   [1,0,0,0,0,0]

        """
        input_ids = []
        phrase_ids = []  # list[list[int]]
        for phrases in phrases_list:
            output = self.encode(phrases)
            _input_ids, _phrase_ids = output["input_ids"], output["phrase_ids"]
            input_ids.append(_input_ids)
            phrase_ids.append(_phrase_ids)
        padded_input_ids = self.lm_tokenizer.pad(
            {"input_ids": input_ids},  # 输入格式要求为字典（可包含多个键）
            padding=True,  # 填充到批次最长序列
            return_tensors="pt",  # 返回PyTorch张量
            return_attention_mask=True,  # 自动生成attention mask
        )
        input_ids = padded_input_ids["input_ids"]
        attention_mask = padded_input_ids["attention_mask"]
        sum = 0
        for i in range(len(phrases_list)):
            input_ids[i] = torch.where((input_ids[i] >= self.static_vocab), input_ids[i] + sum, input_ids[i])
            sum += len(phrase_ids[i])
        combined_phrase_ids = []
        for i in range(len(phrase_ids)):
            combined_phrase_ids.extend(phrase_ids[i])
        padded_phrase_ids = self.phrase_tokenizer.pad(
            {
                "input_ids": combined_phrase_ids,  # 输入格式要求为字典（可包含多个键）
            },
            padding=True,  # 填充到批次最长序列
            return_tensors="pt",  # 返回PyTorch张量
            return_attention_mask=True,  # 自动生成attention mask
        )
        outputs = {
            "input_ids": input_ids,
            "phrase_ids": padded_phrase_ids["input_ids"],
            "attention_mask": attention_mask,
            "phrase_attention_mask": padded_phrase_ids["attention_mask"],
        }
        if logits_mask:
            outputs["mask_ids"] = []
            mask = list(range(self.static_vocab, self.static_vocab + sum))
            now = 0
            for _phrase_ids in phrase_ids:
                outputs["mask_ids"].append(mask[:now] + mask[now + len(_phrase_ids) :])
                now += len(_phrase_ids)
        return outputs

    def decode(
        self, input_ids: list[int], phrases_ids: list[list[int]] = None, return_ids: bool = False, **kwargs
    ) -> dict:
        token_ids = []
        token_phrase_ids = []
        for id in input_ids:
            if id < self.static_vocab:
                token_ids.append(id)
                token_phrase_ids.append(id)
            else:
                phrase_index = id - self.static_vocab
                token_phrase_ids.append(
                    [
                        token_id
                        for token_id in phrases_ids[phrase_index]
                        if token_id != self.phrase_tokenizer.pad_token_id
                    ]
                )
                assert phrase_index < len(phrases_ids), "Invalid phrase index"
                phrase = self.phrase_tokenizer.decode(phrases_ids[phrase_index], skip_special_tokens=True)
                token_ids.extend(self.lm_tokenizer.encode(phrase, add_special_tokens=False))

        decoded_sentence = self.lm_tokenizer.decode(token_ids, skip_special_tokens=True)

        return {
            "decoded_sentence": decoded_sentence,
            "ids": token_phrase_ids if return_ids else None,
        }

    def save_pretrained(self, save_directory: str):
        self.lm_tokenizer.save_pretrained(f"{save_directory}/lm_tokenizer")
        self.phrase_tokenizer.save_pretrained(f"{save_directory}/phrase_tokenizer")

    def update_dv(self, dv: dict[str, list[int]]):
        pass
