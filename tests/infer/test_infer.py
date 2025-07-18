import os
import sys


sys.path.append("/home/jhkuang/projects/DVAGen_related/DVAGen")


from src.dvagen.configs import get_infer_args
from src.dvagen.infer.infer import infer, prepare


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == "__main__":
    infer_args = get_infer_args()
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
        visualize=True,
    )

    print(res)
