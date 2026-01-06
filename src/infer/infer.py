import os.path

import torch
from transformers import AutoTokenizer, LogitsProcessorList

from ..configs.model_args import PhraseSamplerArguments
from ..models.modeling_dva import DVALogitsProcessor, DVAModel
from ..models.phrase import BasePhraseSampler, NTokenPhraseSampler
from ..models.tokenization_dva import DVATokenizer
from .retriever import BaseRetriever, FAISSRetriever


def prepare(
    dva_model_path: str,
    retriever_embedding_model_path: str,
    phrase_encoder_batch_size: int = 64,
    lm_tokenizer_path: str = None,
    phrase_tokenizer_path: str = None,
    retriever_data_file: str = None,
    retriever_vector_store_path: str = None,
    retriever_save_vector_store_path: str = None,
) -> tuple:
    if lm_tokenizer_path is None:
        lm_tokenizer_path = os.path.join(dva_model_path, "lm_tokenizer")
    if phrase_tokenizer_path is None:
        phrase_tokenizer_path = os.path.join(dva_model_path, "phrase_tokenizer")

    # DVAModel
    model = DVAModel.from_pretrained(
        dva_model_path, device_map="auto", phrase_encoder_batch_size=phrase_encoder_batch_size
    )
    model.eval()

    # Phrase Sampler
    phrase_sampler_config = PhraseSamplerArguments(
        phrase_sampler_type="n_tokens",
        fmm_config=None,  # Replace with actual FMMConfig if needed
        n_words_config=None,  # Replace with actual NWordsConfig if needed
        n_tokens_config=None,  # Replace with actual NTokensConfig if needed
    )
    phrase_sampler = NTokenPhraseSampler(
        tokenizer=AutoTokenizer.from_pretrained(phrase_tokenizer_path),
        config=phrase_sampler_config,
    )

    # Tokenizer
    tokenizer = DVATokenizer(
        phrase_sampler_config=phrase_sampler_config,
        model_name_or_path=lm_tokenizer_path,
        phrase_encoder_name_or_path=phrase_tokenizer_path,
        static_vocab=model.vocab_size,
    )
    tokenizer.lm_tokenizer.padding_side = "left"  # We set the padding side to left during inference

    retriever = FAISSRetriever(
        embedding_model_path=retriever_embedding_model_path,
        data_file=retriever_data_file,
        vector_store_path=retriever_vector_store_path,
        save_vector_store_path=retriever_save_vector_store_path,
    )

    return model, phrase_sampler, tokenizer, retriever


@torch.no_grad()
def infer(
    model: DVAModel,
    phrase_sampler: BasePhraseSampler,
    tokenizer: DVATokenizer,
    retriever: BaseRetriever,
    queries: list[str],
    doc_top_k: int,
    return_ids: bool = False,
    **kwargs,
):
    supporting_documents_list = [retriever.retrieve_documents(query, doc_top_k) for query in queries]
    phrase_candidates_list = [
        [phrase for document in documents for phrase in phrase_sampler.sample(document)]
        for documents in supporting_documents_list
    ]
    phrase_inputs = tokenizer.batch_encode(phrase_candidates_list, logits_mask=True)
    prefix_inputs = tokenizer.lm_tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512)

    input_ids = prefix_inputs["input_ids"].to(model.device)
    attention_mask = prefix_inputs["attention_mask"].to(model.device)
    phrase_ids = phrase_attention_mask = None
    if len(phrase_inputs["phrase_ids"]):
        phrase_ids = phrase_inputs["phrase_ids"].to(model.device)
        phrase_attention_mask = phrase_inputs["phrase_attention_mask"].to(model.device)
    mask_phrase_ids = phrase_inputs["mask_ids"]

    dva_embeds = model.get_dva_embeddings(phrase_ids, phrase_attention_mask)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        dva_embeds=dva_embeds,
        logits_processor=LogitsProcessorList([DVALogitsProcessor(mask_phrase_ids)]),
        **kwargs,
    )
    if phrase_ids is not None:
        phrase_ids = phrase_ids.tolist()
    return [tokenizer.decode(output.tolist(), phrase_ids, return_ids=return_ids) for output in outputs]
