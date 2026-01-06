import logging
from dataclasses import dataclass, field

import torch

# from datasets import Dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from src.configs.data_args import DataArguments
from src.configs.model_args import PhraseSamplerArguments

from ..models.tokenization_dva import DVATokenizer


logger = logging.getLogger(__name__)


class DVADataset(Dataset):
    def __init__(
        self,
        model_name_or_path: str,
        phrase_encoder_name_or_path: str,
        phrase_sampler_config: PhraseSamplerArguments,
        data_config: DataArguments,
        base_data: str,
        static_vocab: int,
        **args,
    ):
        # TODO: optimize dvatokenizer implementation
        self.tokenizer = DVATokenizer(
            model_name_or_path=model_name_or_path,
            phrase_encoder_name_or_path=phrase_encoder_name_or_path,
            phrase_sampler_config=phrase_sampler_config,
            static_vocab=static_vocab,
        )

        # self.lm_tokenizer.vocab_size = self.tokenizer.lm_tokenizer.vocab_size
        # TODO: optimize data process
        self.dataset = []
        # grouped_text = ""
        # texts = []
        # with open(data_config.train_file, "r") as file:
        #     lines = file.readlines()
        # for line in lines:
        #     line = line.split("\t")[0].strip()  # Assuming tab-separated values
        #     if line:
        #         texts.append(line)

        # phrase_cache = []  # token = length-1 phrase

        # while len(texts) > 0:
        #     # print("Processing text: ", len(texts))
        #     while len(phrase_cache) < data_config.max_seq_length:
        #         if not texts:
        #             break
        #         text = texts.pop(0)
        #         phrases = self.tokenizer.tokenize(" " + text)
        #         phrase_cache.extend(phrases)
        #         # tokenids, phrase_ids = self.tokenizer.encode(" "+text, add_special_tokens=False)
        #     if len(phrase_cache) < data_config.max_seq_length:
        #         break
        #     phrase_chunk = phrase_cache[: data_config.max_seq_length]
        #     phrase_cache = phrase_cache[data_config.max_seq_length :]
        #     # phrases = self.
        #     encoded_chunk = self.tokenizer.encode(phrase_chunk, add_special_tokens=False)
        #     input_ids, phrase_ids = encoded_chunk["input_ids"], encoded_chunk["phrase_ids"]
        #     self.dataset.append(
        #         {
        #             "input_ids": input_ids,
        #             "phrase_ids": phrase_ids,
        #         }
        #     )
        texts = []
        with open(data_config.train_file, "r") as file:
            lines = file.readlines()
        for line in lines:
            line = line.split("\t")[0].strip()  # Assuming tab-separated values
            if line:
                texts.append(line)

        token_cache = []  # token = length-1 phrase
        tokenized_text_cache = []
        while len(texts) > 0:
            while len(token_cache) < data_config.max_seq_length:
                if not texts:
                    break
                text = texts.pop(0)
                tokens = self.tokenizer.lm_tokenizer.encode(" " + text, add_special_tokens=False)
                tokenized_text = self.tokenizer.lm_tokenizer.tokenize(" " + text)
                token_cache.extend(tokenized_text)
                tokenized_text_cache.extend(tokens)
                # tokenids, phrase_ids = self.tokenizer.encode(" "+text, add_special_tokens=False)
            if len(token_cache) < data_config.max_seq_length:
                break
            self.dataset.append(
                {
                    "tokens": token_cache[: data_config.max_seq_length],
                    "text": self.tokenizer.lm_tokenizer.decode(
                        tokenized_text_cache[: data_config.max_seq_length], skip_special_tokens=True
                    ),
                }
            )
            token_cache = token_cache[data_config.max_seq_length :]
            tokenized_text_cache = tokenized_text_cache[data_config.max_seq_length :]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        phrases = self.tokenizer.tokenize(self.dataset[idx]["text"])
        return phrases


if __name__ == "__main__":
    pass
    # Example usage
    # data_config = DataArguments(
    #     train_file="/home/weidu/DVAGen/tests/data/base_data_128.txt",
    #     validation_file="path/to/validation.json",
    #     max_seq_length=128,
    #     max_train_samples=1000,
    #     max_eval_samples=500,
    #     overwrite_cache=True,
    # )
    # phrase_sampler_config = PhraseSamplerArguments(
    #     phrase_sampler_type="n_tokens",
    #     fmm_config=None,  # Replace with actual FMMConfig if needed
    #     n_words_config=None,  # Replace with actual NWordsConfig if needed
    #     n_tokens_config=None,  # Replace with actual NTokensConfig if needed
    # )
    # model_config = BaseModelArguments(
    #     model_name_or_path="/home/weidu/public/pretrain/Qwen/Qwen3-0.6B",
    #     phrase_encoder_name_or_path="/home/weidu/public/old_pretrain/gpt2",
    #     backbone_name_or_path="bert-base-uncased",
    #     backbone_config_file="path/to/backbone/config.json",
    # )

    # dataset = DVADataset(
    #     model_name_or_path="/home/weidu/public/pretrain/Qwen/Qwen3-0.6B",
    #     phrase_encoder_name_or_path="/home/weidu/public/old_pretrain/gpt2",
    #     phrase_sampler_config=phrase_sampler_config,
    #     data_config=data_config,
    #     base_data="path/to/base/data",
    # )

    # train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn)
    # print(next(iter(train_dataloader)))
    #     for item in tqdm(dataset):
    #         print(item)
    # training_set = DVADataset(
    #     model_name_or_path=train_args.model.language_model_path,
    #     phrase_encoder_name_or_path=train_args.model.phrase_encoder_path,
    #     phrase_sampler_config=phrase_sampler_config,
    #     data_config=data_config,
    #     base_data="path/to/base/data",
    # )
