from transformers import AutoConfig, PretrainedConfig


class DVAConfig(PretrainedConfig):
    model_type = "dva"

    def __init__(
        self,
        language_model_config: PretrainedConfig | dict | None = None,
        phrase_encoder_config: PretrainedConfig | dict | None = None,
        use_phrase_encoder_proj: bool = True,
        phrase_encoder_proj_pdrop: float = 0.1,
        phrase_encoder_proj_act: str = "relu",
        phrase_encoder_batch_size: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # When using `from_pretrained()`, both `language_model_config` and `phrase_encoder_config` are loaded as
        # `dict` types. To properly load the configs with `AutoModel.from_config()`, we need to manually convert them
        # into instances of their respective `PretrainedConfig` subclasses. Note that the configuration should be a
        # specific subclass of `PretrainedConfig`, not the base class itself.

        self.language_model_config = self._set_model_config(language_model_config)
        self.phrase_encoder_config = self._set_model_config(phrase_encoder_config)
        # self.language_model_config = language_model_config
        # self.phrase_encoder_config = phrase_encoder_config

        # Whether to use a projection layer on the phrase encoder outputs.
        self.use_phrase_encoder_proj = use_phrase_encoder_proj

        # Dropout probability for the phrase encoder projection layer.
        # Not used if `use_phrase_encoder_proj` is False.
        self.phrase_encoder_proj_pdrop = phrase_encoder_proj_pdrop

        # Activation function for the phrase encoder projection layer.
        # Not used if `use_phrase_encoder_proj` is False.
        self.phrase_encoder_proj_act = phrase_encoder_proj_act

        # Batch size for the phrase encoder. Helps prevent OOM.
        # TODO Remove it from the config
        self.phrase_encoder_batch_size = phrase_encoder_batch_size

    @staticmethod
    def _set_model_config(config: PretrainedConfig | dict | None = None):
        if isinstance(config, dict):
            config = AutoConfig.for_model(**config)
        return config

    @classmethod
    def to_dataclass(cls):
        import inspect
        from dataclasses import field, make_dataclass
        from typing import Any

        sig = inspect.signature(cls.__init__)
        fields = []
        for name, param in sig.parameters.items():
            if name in ("self", "args", "kwargs"):
                continue
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any
            if param.default == inspect.Parameter.empty:
                fields.append((name, annotation))
            else:
                fields.append((name, annotation, field(default=param.default)))
        DataCls = make_dataclass(f"{cls.__name__}Data", fields)
        return DataCls
