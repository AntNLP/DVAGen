import os
import sys


sys.path.append("/home/jhkuang/projects/DVAGen_related/DVAGen")

import simple_parsing

from dvagen.configs.parser import InferArgs
from dvagen.infer.infer import infer, prepare


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == "__main__":
    config_path = (
        "/home/jhkuang/projects/DVAGen_related/DVAGen/examples/chat.yaml"
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
    res = infer(
        model=model,
        phrase_sampler=phrase_sampler,
        tokenizer=tokenizer,
        retriever=retriever,
        queries=[
            "Introduce China to me:",
            "Introduce US to me:",
            "Introduce Canada to me:",
        ],
        doc_top_k=32,
        # visualize=True,
    )

    print(res)
