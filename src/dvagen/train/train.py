import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoTokenizer

from ..configs.model_args import PhraseSamplerType
from ..configs.parser import TrainArgs
from ..configs.train_args import FinetuningType
from ..datasets.dvadataset import DVADataset
from ..models.configuration_dva import DVAConfig
from ..models.modeling_dva import DVAModel
from ..models.sampler import FMMPhraseSampler, NTokenPhraseSampler, NWordsPhraseSampler
from ..models.tokenization_dva import DVATokenizer
from ..utils import logging
from .trainer import DVATrainer


logger = logging.get_logger(__name__)


class DVACollator:
    def __init__(self, tokenizer: DVATokenizer, device, params=None):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        outputs = self.tokenizer.batch_encode(batch, phrases_mask=False)
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

    model = DVAModel(model_config)
    model.initialize_modules(
        language_model_path=train_args.model.language_model_path,
        phrase_encoder_path=train_args.model.phrase_encoder_path,
    )
    if train_args.train.finetuning_type == FinetuningType.LORA.value:
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
    elif train_args.train.finetuning_type == FinetuningType.FULL.value:
        if train_args.train.lora is not None:
            logger.warning_rank0(
                "LoRA arguments are provided but finetuning type is set to 'full'. Ignoring LoRA settings."
            )
    elif train_args.train.finetuning_type == FinetuningType.FREEZE.value:
        logger.info_rank0("The language model is frozen during training.")
        for param in model.language_model.parameters():
            param.requires_grad_(False)
    else:
        raise ValueError(f"Unsupported finetuning type: {train_args.train.finetuning_type}")

    # TODO In the future version, PhraseSamplerConfig will be used instead.
    if train_args.model.phrase_sampler_type == PhraseSamplerType.N_TOKENS:
        phrase_tokenizer = AutoTokenizer.from_pretrained(train_args.model.sampler_model_path)
        sampler = NTokenPhraseSampler(
            tokenizer=phrase_tokenizer,
            random_up=train_args.model.sampler_random_up,
            random_low=train_args.model.sampler_random_low,
            phrase_max_length=train_args.model.phrase_max_length,
        )
    elif train_args.model.phrase_sampler_type == PhraseSamplerType.N_WORDS:
        sampler = NWordsPhraseSampler(
            random_up=train_args.model.sampler_random_up,
            random_low=train_args.model.sampler_random_low,
            phrase_max_length=train_args.model.phrase_max_length,
        )
    elif train_args.model.phrase_sampler_type == PhraseSamplerType.FMM:
        sampler = FMMPhraseSampler(
            ignore_first=train_args.model.ignore_first,
            embedding_model_path=train_args.model.fmm_embedding_model_path,
            data_file=train_args.model.fmm_data_file,
            vector_store_path=train_args.model.fmm_vector_store_path,
            min_length=train_args.model.fmm_min_length,
            max_length=train_args.model.fmm_max_length,
        )

    tokenizer = DVATokenizer(
        model_name_or_path=train_args.model.language_model_path,
        phrase_encoder_name_or_path=train_args.model.phrase_encoder_path,
        phrase_sampler_type=train_args.model.phrase_sampler_type,
        static_vocab=model.vocab_size,
        sampler=sampler,
    )
    training_set = DVADataset(
        model_name_or_path=train_args.model.language_model_path,
        phrase_encoder_name_or_path=train_args.model.phrase_encoder_path,
        static_vocab=model.vocab_size,
        tokenizer=tokenizer,
        data_path=train_args.data.train_path,
        save_data_path=train_args.data.save_train_path,
        data_source=train_args.data.data_source,
        max_seq_length=train_args.data.max_seq_length,
    )
    collator = DVACollator(tokenizer=tokenizer, device=train_args.train.device)
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
