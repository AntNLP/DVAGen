# Details of the configuration
Different commands need different argument sets.
- chat: infer, model
- train: train, data, model
- eval: eval, infer, model 

## eval
- `test_data_file(str)`: The data path for evaluation.
- `batch_size(int)`: The batch size of evaluation for one step.
- `task_type(EvalTaskType)`: The evaluation task type.(more details in [eval_args.py](../src/dvagen/configs/eval_args.py) )
- `eval_seed(int)`: The ramdom seed for evaluation.
- `save_results_path(str)`: The path for saving results.
- `prefix_tokenizer_path(str): The path for the tokenizer used to extract input prefix tokens.
- `prefix_tokens(int)`: Maximum prefix tokens.
- `mauve_model_path(str)`: The path for the model used to obtain features when computing the MAUVE metric.
- `mauve_batch_size(int)`: The batch size of the model.
- `perplexity_model_path(str)`: The path for the model used to compute the Perplexity metric.
- `perplexity_batch_size(int)`: The batch size of the model.
- `nsl_tokenizer_path(str)`: The path for the tokenizer used to compute the NSL metric.

## data
- `data_source(str)`: The source of data(eg. "wikitext103", "fineweb").
- `train_path(str)`: The path of training data.
- `save_train_path(str)`: The path of processed training data. If `train_path` is "None", model will load data from this path.
- `validation_path(str)`: The path of validation data.
- `save_validation_path(str)`: The path of processed training data. If `validation_path` is "None", model will load data from this path.
- `max_seq_length(int)`: The max sequence length for one example.

## model
- `phrase_sampler_type(PhraseSamplerType)`: The type of phrase sampler including `FMM`, `N_TOKENS`, `N_WORDS`.(more details in [model_args.py](../src/dvagen/configs/model_args.py))
- `sampler_model_path(str)`: The model path for phrase sampler to tokenize text.
- `sampler_random_up(int)`: The max length of phrase gap.
- `sampler_random_low(int)`: The min length of phrase gap.
- `phrase_max_length(int)`: The max length of phrase.

The FMM sampler follows the implemention in [COG](https://github.com/gmftbyGMFTBY/Copyisallyouneed/blob/main/data/wikitext103_1024/phrase_split/phrase_split.py)
- `fmm_embedding_model_path(str)`: The embedding model for faiss index.
- `fmm_data_file(str)`: The retrieval source of FMM.
- `fmm_vector_store_path(str)`: The path of the vector store index.
- `fmm_save_vector_store_path(str)`: The save path of the vector store index.
- `fmm_min_length(int)`: The min length for FMM. 
- `fmm_max_length(int)`: The max length for FMM. 

## infer

- `doc_top_k(int)`: The top K supporting documents to retrieve for each query.
- `embedding_model_path(str)`: The path for the embedding model used in retrieval.
- `data_file(str)`: The data path of the supporting documents.
- `vector_store_path(str)`: The path of the vector store index.
- `save_vector_store_path(str)`: The save path of the vector store index.

The same setting as `generate` method in huggingface.
- `do_sample(bool)`: Whether or not to use sampling ; use greedy decoding otherwise.
- `temperature(float)`: The value used to module the next token probabilities. This value is set in a model's `generation_config.json` file. If it isn't set, the default value is 1.0
- `max_length(int)`: The maximum length the generated tokens can have. Corresponds to the length of the input prompt + `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
- `max_new_tokens(int)`: The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
- `top_k(int)`: The number of highest probability vocabulary tokens to keep for top-k-filtering. This value is set in a model's `generation_config.json` file. If it isn't set, the default value is 50.
- `top_p(float)`:If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation. This value is set in a model's `generation_config.json` file. If it isn't set, the default value is 1.0

## train

- Includes same arguments as huggingface TrainingArguments.
- `finetuning_type`: Training method for model including "freeze"(freeze backbone model), "lora", "full"(full finetune).
- `r`: Lora attention dimension (the “rank”).
- `alpha`: The alpha parameter for Lora scaling.
- `dropout`: The dropout probability for Lora layers.
- `target_modules`: The names of the modules to apply the adapter to. If this is specified, only the modules with the specified names will be replaced. When passing a string, a regex match will be performed. When passing a list of strings, either an exact match will be performed or it is checked if the name of the module ends with any of the passed strings. If this is specified as ‘all-linear’, then all linear/Conv1D modules are chosen (if the model is a PreTrainedModel, the output layer excluded). If this is not specified, modules will be chosen according to the model architecture. If the architecture is not known, an error will be raised — in this case, you should specify the target modules(the same as LoraConfig in huggingface)

