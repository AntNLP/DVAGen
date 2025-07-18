import json
import os

from tqdm.auto import tqdm
from transformers import AutoTokenizer, set_seed

from ..configs.eval_args import EvalTaskType
from ..configs.parser import EvalArgs
from ..infer.infer import infer, prepare
from .metrics import BytesPerTokenMetric, MauveMetric, MetricList, NSLMetric, PerplexityMetric, RepMetric


def predict_results(
    eval_args: EvalArgs,
) -> list[dict[str, str]]:
    model, phrase_sampler, tokenizer, retriever = prepare(
        dva_model_path=eval_args.model.model_name_or_path,
        retriever_embedding_model_path=eval_args.infer.embedding_model_path,
        phrase_encoder_batch_size=eval_args.model.phrase_encoder_batch_size,
        lm_tokenizer_path=eval_args.model.language_model_path,
        phrase_tokenizer_path=eval_args.model.phrase_encoder_path,
        retriever_data_file=eval_args.infer.data_file,
        retriever_vector_store_path=eval_args.infer.vector_store_path,
        retriever_save_vector_store_path=eval_args.infer.save_vector_store_path,
        phrase_sampler_type=eval_args.model.phrase_sampler_type,
        sampler_model_path=eval_args.model.sampler_model_path,
        sampler_random_up=eval_args.model.sampler_random_up,
        sampler_random_low=eval_args.model.sampler_random_low,
        phrase_max_length=eval_args.model.phrase_max_length,
        fmm_embedding_model_path=eval_args.model.fmm_embedding_model_path,
        fmm_data_file=eval_args.model.fmm_data_file,
        fmm_vector_store_path=eval_args.model.fmm_vector_store_path,
        fmm_save_vector_store_path=eval_args.model.fmm_save_vector_store_path,
        fmm_min_length=eval_args.model.fmm_min_length,
        fmm_max_length=eval_args.model.fmm_max_length,
    )

    with open(eval_args.eval.test_data_file) as f:
        test_data = f.readlines()

    prefix_tokenizer = AutoTokenizer.from_pretrained(eval_args.eval.prefix_tokenizer_path)

    references = []
    predictions = []
    for query in tqdm(range(0, len(test_data), eval_args.eval.batch_size)):
        batch_queries = [q.strip() for q in test_data[query : query + eval_args.eval.batch_size]]
        references.extend(batch_queries)
        batch_prefixes = [
            prefix_tokenizer.decode(
                prefix_tokenizer.encode(q)[: eval_args.eval.prefix_tokens], skip_special_tokens=True
            )
            for q in batch_queries
        ]
        batch_outputs = infer(
            model,
            phrase_sampler,
            tokenizer,
            retriever,
            return_ids=True,
            queries=batch_prefixes,
            doc_top_k=eval_args.infer.doc_top_k,
            do_sample=eval_args.infer.do_sample,
            temperature=eval_args.infer.temperature,
            top_k=eval_args.infer.top_k,
            max_new_tokens=eval_args.infer.max_new_tokens,
        )
        predictions.extend(batch_outputs)

    predicted_results = [
        {"prediction": pred["decoded_sentence"], "reference": ref, "ids": pred["ids"]}
        for pred, ref in zip(predictions, references)
    ]

    if eval_args.eval.save_results_path is not None:
        with open(eval_args.eval.save_results_path, "w") as f:
            json.dump(predicted_results, f)

    return predicted_results


def report_metrics(predicted_results: list[dict[str, str | list]], eval_args: EvalArgs) -> dict[str, float]:
    predictions = [result["prediction"] for result in predicted_results]
    references = [result["reference"] for result in predicted_results]
    predictions_ids = [result["ids"] for result in predicted_results]

    if eval_args.eval.task_type == EvalTaskType.LANGUAGE_MODELING:
        eval_metrics = MetricList(
            [
                MauveMetric(
                    predictions=predictions,
                    references=references,
                    model_name_or_path=eval_args.eval.perplexity_model_path,
                    batch_size=eval_args.eval.mauve_batch_size,
                ),
                RepMetric(
                    predictions=predictions,
                ),
                PerplexityMetric(
                    predictions=predictions,
                    model_name_or_path=eval_args.eval.perplexity_model_path,
                    batch_size=eval_args.eval.perplexity_batch_size,
                ),
                NSLMetric(
                    predictions=predictions,
                    predictions_ids=predictions_ids,
                    model_name_or_path=eval_args.eval.nsl_tokenizer_path,
                ),
                BytesPerTokenMetric(
                    predictions=predictions,
                    predictions_ids=predictions_ids,
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported eval task type: {eval_args.eval.task_type}")

    return eval_metrics.compute()


def evaluate(eval_args: EvalArgs):
    if eval_args.eval.eval_seed is not None:
        set_seed(eval_args.eval.eval_seed)

    if eval_args.eval.save_results_path is not None and os.path.exists(eval_args.eval.save_results_path):
        with open(eval_args.eval.save_results_path) as f:
            predicted_results = json.load(f)
        return report_metrics(predicted_results, eval_args)

    predicted_results = predict_results(eval_args)
    return report_metrics(predicted_results, eval_args)
