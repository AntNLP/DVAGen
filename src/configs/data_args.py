from dataclasses import dataclass


@dataclass
class DataArguments:
    train_file: str
    validation_file: str
    max_seq_length: int = 1024
    max_train_samples: int = -1
    max_eval_samples: int = -1
    overwrite_cache: bool = False
