import logging

import datasets
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from ..models.tokenization_dva import DVATokenizer


logger = logging.getLogger(__name__)


class DVADataset(Dataset):
    def __init__(
        self,
        tokenizer: DVATokenizer,
        data_path: str = None,
        save_data_path: str = None,
        max_seq_length: int = 512,
        data_source: str = "wikitext103",
        cut_len: int = -1,
        **kwargs,
    ):
        # TODO: optimize data process
        self.tokenizer = tokenizer
        if data_path is not None and data_path != "None":
            # Load the dataset from the saved path
            self.dataset = data_process(
                data_file=data_path,
                data_source=data_source,
                max_seq_length=max_seq_length,
                tokenizer=tokenizer.lm_tokenizer,
            )
            if save_data_path is not None:
                data_save(self.dataset, save_path=save_data_path)
        else:
            self.dataset = data_load(save_path=save_data_path)
        self.dataset = self.dataset.select(range(cut_len)) if cut_len > 0 else self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        phrases = self.tokenizer.tokenize(self.dataset[idx]["text"])
        return phrases


def data_process(data_file: str, data_source: str, max_seq_length: int, tokenizer: AutoTokenizer, **kwargs):
    """
    Process the data for DVADataset.
    This function is a placeholder and should be implemented based on specific requirements.
    """
    tokenizer_name = tokenizer.name_or_path.split("/")[-1]
    if data_source == "wikitext103":
        texts = []
        with open(data_file) as file:
            lines = file.readlines()
        for line in lines:
            line = line.split("\t")[0].strip()  # Assuming tab-separated values
            if line:
                texts.append(line)
        # TODO 保存使用DVATokenizer.lm_tokenizer tokenized之后的结果
        token_cache = []  # token = length-1 phrase
        dataset = []
        while len(texts) > 0:
            while len(token_cache) < max_seq_length:
                if not texts:
                    break
                text = texts.pop(0)
                tokens = tokenizer.tokenize(" " + text)
                token_cache.extend(tokens)
                # tokenids, phrase_ids = self.tokenizer.encode(" "+text, add_special_tokens=False)
            if len(token_cache) < max_seq_length:
                break
            dataset.append(
                {
                    "text": tokenizer.convert_tokens_to_string(
                        token_cache[:max_seq_length]
                    ),  # Convert tokens back to string for text representation
                    # "tokens": token_cache[:max_seq_length],
                }
            )
            token_cache = token_cache[max_seq_length:]
        dataset = datasets.Dataset.from_list(dataset)
    elif data_source == "fineweb":
        # for sub_set in ["000_00000.parquet", "001_00000.parquet"]:
        token_cache = []
        dataset = []
        total_tokens = 0
        pbar = tqdm(total=1000000000, desc="Tokenizing data")
        for sub_set in ["000_00000.parquet", "001_00000.parquet", "002_00000.parquet", "003_00000.parquet"]:
            _dataset = datasets.load_dataset(
                "parquet",
                data_files=f"{data_file}/{sub_set}",
                split="train",
            )
            for example in _dataset:
                text = example["text"]
                if example["language"] != "en":
                    continue
                if len(token_cache) == 0:
                    tokens = tokenizer.tokenize(text)
                else:
                    tokens = tokenizer.tokenize("\n" + text)
                if len(tokens) > 131050:
                    continue
                token_cache.extend(tokens)
                while len(token_cache) >= max_seq_length:
                    dataset.append(
                        {
                            "text": tokenizer.convert_tokens_to_string(token_cache[:max_seq_length]),
                        }
                    )
                    token_cache = token_cache[max_seq_length:]
                    total_tokens += max_seq_length
                    pbar.update(max_seq_length)
            if pbar.n >= 1000000000:
                break
        dataset = datasets.Dataset.from_list(dataset)
        pbar.close()
    return dataset


def data_save(dataset: datasets.Dataset, save_path: str):
    """
    Save the processed data to a specified path.
    This function is a placeholder and should be implemented based on specific requirements.
    """
    # Example implementation
    dataset.to_json(save_path, orient="records", lines=True)
    return


def data_load(save_path: str):
    return datasets.load_dataset("json", data_files=save_path, split="train")
