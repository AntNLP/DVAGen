import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig

from src.datasets.dvadataset import DVADataset
from src.models.tokenization_dva import DVATokenizer
from src.train.trainer import DVATrainer

from ..configs.data_args import DataArguments
from ..configs.model_args import PhraseSamplerArguments
from ..configs.parser import TrainArgs
from ..configs.train_args import FinetuningType
from ..models.configuration_dva import DVAConfig
from ..models.modeling_dva import DVAModel
from ..utils import logging


logger = logging.get_logger(__name__)


class DVACollator:
    def __init__(self, tokenizer: DVATokenizer, device, params=None):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        outputs = self.tokenizer.batch_encode(batch, logits_mask=False)
        outputs["labels"] = torch.where(outputs["attention_mask"] == 1, outputs["input_ids"], torch.tensor(-100))
        return outputs


def train(train_args: TrainArgs):
    model_config = DVAConfig(
        language_model_config=AutoConfig.from_pretrained(train_args.model.language_model_path),
        phrase_encoder_config=AutoConfig.from_pretrained(train_args.model.phrase_encoder_path),
        use_phrase_encoder_proj=train_args.model.use_phrase_encoder_proj,
        phrase_encoder_proj_pdrop=train_args.model.phrase_encoder_proj_pdrop,
        phrase_encoder_proj_act=train_args.model.phrase_encoder_proj_act,
        phrase_encoder_batch_size=train_args.model.phrase_encoder_batch_size,
    )

    data_config = DataArguments(
        train_file=train_args.data.train_file,
        validation_file=train_args.data.validation_file,
        max_seq_length=512,
        max_train_samples=1000000,
        max_eval_samples=500,
        overwrite_cache=True,
    )
    phrase_sampler_config = PhraseSamplerArguments(
        phrase_sampler_type="n_tokens",
        fmm_config=None,  # Replace with actual FMMConfig if needed
        n_words_config=None,  # Replace with actual NWordsConfig if needed
        n_tokens_config=None,  # Replace with actual NTokensConfig if needed
    )

    model = DVAModel(model_config)
    model.initialize_modules(
        language_model_path=train_args.model.language_model_path,
        phrase_encoder_path=train_args.model.phrase_encoder_path,
    )

    if train_args.train.finetuning_type == FinetuningType.LORA:
        assert train_args.train.lora is not None, "LoRA arguments must be provided for LoRA fine-tuning."
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=train_args.train.lora.r,
            lora_alpha=train_args.train.lora.alpha,
            lora_dropout=train_args.train.lora.dropout,
            target_modules=train_args.train.lora.target_modules,
        )
        model = get_peft_model(model, peft_config)
    elif train_args.train.finetuning_type == FinetuningType.FULL:
        if train_args.train.lora is not None:
            logger.warning_rank0(
                "LoRA arguments are provided but finetuning type is set to 'full'. Ignoring LoRA settings."
            )
    elif train_args.train.finetuning_type == FinetuningType.FREEZE:
        logger.info_rank0("The language model is frozen during training.")
        for param in model.language_model.parameters():
            param.requires_grad_(False)
    else:
        raise ValueError(f"Unsupported finetuning type: {train_args.train.finetuning_type}")

    # training_set = DVAtestDataset(sv_vocab_size=model.config.language_model_config.vocab_size)
    training_set = DVADataset(
        model_name_or_path=train_args.model.language_model_path,
        phrase_encoder_name_or_path=train_args.model.phrase_encoder_path,
        phrase_sampler_config=phrase_sampler_config,
        data_config=data_config,
        base_data="path/to/base/data",
        static_vocab=model.vocab_size,
    )
    collator = DVACollator(tokenizer=training_set.tokenizer, device=train_args.train.device)
    trainer = DVATrainer(
        model=model.cuda(),
        args=train_args.train,
        train_dataset=training_set,
        data_collator=collator,
    )
    if train_args.train.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=train_args.train.resume_from_checkpoint)
    else:
        trainer.train()
    trainer.save_state()
