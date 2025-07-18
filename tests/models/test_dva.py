import torch
import pytest

from src.models.modeling_dva import DVAConfig, DVAModel


test_configs = [
    {
        "language_model_path": "../public/old_pretrain/gpt2",
        "phrase_encoder_path": "../public/old_pretrain/gpt2",
        "static_vocab_size": 50257,
        "dynamic_vocab_size": 256,
    },
    {
        "language_model_path": "../public/pretrain/meta-llama/Llama-3.2-1B-Instruct",
        "phrase_encoder_path": "../public/old_pretrain/gpt2",
        "static_vocab_size": 128256,
        "dynamic_vocab_size": 256,
    },
    {
        "language_model_path": "../public/old_pretrain/gpt2",
        "phrase_encoder_path": "../public/pretrain/meta-llama/Llama-3.2-1B-Instruct",
        "static_vocab_size": 50257,
        "dynamic_vocab_size": 256,
    },
]


@pytest.fixture(params=test_configs)
def configs(request):
    return request.param


@pytest.fixture
def dva(configs):
    dva_config = DVAConfig(
        language_model_config=None,
        phrase_encoder_config=None,
        use_phrase_encoder_proj=True,
        phrase_encoder_proj_pdrop=0.1,
        phrase_encoder_proj_act="relu",
        phrase_encoder_batch_size=64,
    )
    model = DVAModel(dva_config)
    model.initialize_modules(
        language_model_path=configs["language_model_path"],
        phrase_encoder_path=configs["phrase_encoder_path"],
    )
    model.to("cuda:0")
    return model


@pytest.fixture
def dummy_inputs(configs):
    static_vocab_size = configs["static_vocab_size"]
    dynamic_vocab_size = configs["dynamic_vocab_size"]

    input_ids = torch.randint(0, static_vocab_size + dynamic_vocab_size, (4, 512))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    phrase_ids = torch.randint(0, 1024, (dynamic_vocab_size, 8))
    phrase_attention_mask = torch.ones_like(phrase_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "phrase_ids": phrase_ids,
        "phrase_attention_mask": phrase_attention_mask,
    }


def test_dva_forward(dva, dummy_inputs, configs):
    inputs = dummy_inputs
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

    outputs = dva(**inputs)
    assert outputs.loss is not None
    assert outputs.logits.shape == (
        dummy_inputs["input_ids"].shape[0],
        dummy_inputs["input_ids"].shape[1],
        configs["static_vocab_size"] + configs["dynamic_vocab_size"],
    )


def test_dva_generate(dva, dummy_inputs):
    inputs = dummy_inputs
    inputs.pop("labels", None)
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

    dva.generate(**inputs)
