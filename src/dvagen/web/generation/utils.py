import os

import simple_parsing
import torch

from dvagen.configs.parser import InferArgs
from dvagen.infer.infer import prepare
from dvagen.models.phrase import Phrase
from dvagen.utils import logging

if "dvagen_root" in os.environ and "config_path" in os.environ:
    logger = logging.get_logger(__name__)
    logger.debug(">>> Lodding DVAGen >>>")
    config_path = os.path.join(
        os.environ["dvagen_root"], os.environ["config_path"]
    )
    infer_args = simple_parsing.parse(
        config_class=InferArgs,
        conflict_resolution=simple_parsing.ConflictResolution.NONE,
        argument_generation_mode=simple_parsing.ArgumentGenerationMode.FLAT,
        args=["--config_path", str(config_path)],
        add_config_path_arg=True,
    )
    model, phrase_sampler, tokenizer, retriever = prepare(
        dva_model_path=infer_args.model.model_name_or_path,
        retriever_embedding_model_path=infer_args.infer.embedding_model_path,
        phrase_encoder_batch_size=infer_args.model.phrase_encoder_batch_size,
        lm_tokenizer_path=infer_args.model.language_model_path,
        phrase_tokenizer_path=infer_args.model.phrase_encoder_path,
        retriever_data_file=infer_args.infer.data_file,
        retriever_vector_store_path=infer_args.infer.vector_store_path,
        retriever_save_vector_store_path=infer_args.infer.save_vector_store_path,
        phrase_sampler_type=infer_args.model.phrase_sampler_type,
        sampler_model_path=infer_args.model.sampler_model_path,
        sampler_random_up=infer_args.model.sampler_random_up,
        sampler_random_low=infer_args.model.sampler_random_low,
        phrase_max_length=infer_args.model.phrase_max_length,
        fmm_embedding_model_path=infer_args.model.fmm_embedding_model_path,
        fmm_data_file=infer_args.model.fmm_data_file,
        fmm_vector_store_path=infer_args.model.fmm_vector_store_path,
        fmm_save_vector_store_path=infer_args.model.fmm_save_vector_store_path,
        fmm_min_length=infer_args.model.fmm_min_length,
        fmm_max_length=infer_args.model.fmm_max_length,
    )
    logger.debug("<<< DVAGen Loaded <<<")


def generate_util(prefix: str, phrases: list[str]) -> list[dict]:
    if len(phrases) == 0:
        supporting_documents_list = [
            retriever.retrieve_documents(query, 32) for query in [prefix]
        ]
        phrases = [
            [
                phrase
                for document in documents
                for phrase in phrase_sampler.sample(document)
            ]
            for documents in supporting_documents_list
        ]
    else:
        phrases = [
            [Phrase(content=phrase, is_phrase=True)] for phrase in phrases
        ]
    phrase_inputs = tokenizer.batch_encode(phrases, phrases_mask=True)
    prefix_inputs = tokenizer.lm_tokenizer(
        [prefix],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    input_ids = prefix_inputs["input_ids"].to(model.device)
    attention_mask = prefix_inputs["attention_mask"].to(model.device)
    phrase_ids = phrase_attention_mask = None
    if len(phrase_inputs["phrase_ids"]):
        phrase_ids = phrase_inputs["phrase_ids"].to(model.device)
        phrase_attention_mask = phrase_inputs["phrase_attention_mask"].to(
            model.device
        )
    mask_phrase_ids = phrase_inputs["mask_ids"]

    with torch.no_grad():
        dva_embeds = model.get_dva_embeddings(phrase_ids, phrase_attention_mask)
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dva_embeds=dva_embeds,
            do_sample=True,
            max_new_tokens=100,
            temperature=0.7,
            top_k=100,
            output_scores=True,
            return_dict_in_generate=True,
        )
    if phrase_ids is not None:
        phrase_ids = phrase_ids.tolist()

    # get suffix
    suffix_ids = generation_output.sequences[0][len(input_ids[0]) :]
    suffix_scores = generation_output.scores

    # print(len(suffix_ids), len(suffix_scores))  # type: ignore

    # Results
    results = []
    for step in range(len(suffix_scores)):  # type: ignore
        # choosen token
        step_id = suffix_ids[step].item()
        step_token = tokenizer.decode(
            [step_id], phrases_ids=phrase_ids, return_ids=False
        )

        # candidate 100 tokens
        step_logits = suffix_scores[step][0]  # type: ignore
        step_prob = torch.softmax(step_logits, dim=-1)
        top_100_probs, tok_100_ids = torch.topk(step_prob, k=100)

        results.append(
            {
                "chosenToken": step_token["decoded_sentence"],
                "type": "token"
                if step_id < tokenizer.static_vocab
                else "phrase",
                "alternatives": [
                    {
                        "token": tokenizer.decode(
                            [idx], phrases_ids=phrase_ids, return_ids=False
                        )["decoded_sentence"],
                        "prob": prob.item(),
                        "type": "token"
                        if idx < tokenizer.static_vocab
                        else "phrase",
                    }
                    for idx, prob in zip(
                        tok_100_ids,
                        top_100_probs,
                    )
                ],
            }
        )
    return results
