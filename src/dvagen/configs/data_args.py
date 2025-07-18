from dataclasses import dataclass


@dataclass
class DataArguments:
    train_path: str
    data_source: str
    validation_path: str
    save_train_path: str = None
    save_validation_path: str = None
    max_seq_length: int = 1024
    max_train_samples: int = -1
    max_eval_samples: int = -1
    overwrite_cache: bool = False
