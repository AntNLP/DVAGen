from torch.utils.data import DataLoader

from src.configs.data_args import DataArguments
from src.configs.model_args import BaseModelArguments, PhraseSamplerArguments
from src.datasets.dvadataset import DVADataset


data_config = DataArguments(
    train_file="/home/weidu/DVAGen/tests/data/50.txt",
    validation_file="path/to/validation.json",
    max_seq_length=128,
    max_train_samples=1000,
    max_eval_samples=500,
    overwrite_cache=True,
)
phrase_sampler_config = PhraseSamplerArguments(
    phrase_sampler_type="n_tokens",
    fmm_config=None,  # Replace with actual FMMConfig if needed
    n_words_config=None,  # Replace with actual NWordsConfig if needed
    n_tokens_config=None,  # Replace with actual NTokensConfig if needed
)
model_config = BaseModelArguments(
    model_name_or_path="/home/weidu/public/pretrain/Qwen/Qwen3-0.6B",
    phrase_encoder_name_or_path="/home/weidu/public/old_pretrain/gpt2",
    backbone_name_or_path="bert-base-uncased",
    backbone_config_file="path/to/backbone/config.json",
)

dataset = DVADataset(
    model_name_or_path="/home/weidu/public/pretrain/Qwen/Qwen3-0.6B",
    phrase_encoder_name_or_path="/home/weidu/public/old_pretrain/gpt2",
    phrase_sampler_config=phrase_sampler_config,
    data_config=data_config,
    base_data="path/to/base/data",
)

train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn)
batch = next(iter(train_dataloader))
print(batch["input_ids"][0])
print(batch["phrase_ids"])  # This should print the phrase IDs for the first item in the batch
print(dataset.tokenizer.decode(batch["input_ids"][0], batch["phrase_ids"]))
