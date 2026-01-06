from ..configs.parser import InferArgs
from .infer import infer, prepare


def chat(infer_args: InferArgs):
    model, phrase_sampler, tokenizer, retriever = prepare(
        dva_model_path=infer_args.model.model_name_or_path,
        retriever_embedding_model_path=infer_args.infer.embedding_model_path,
        phrase_encoder_batch_size=infer_args.model.phrase_encoder_batch_size,
        lm_tokenizer_path=infer_args.model.language_model_path,
        phrase_tokenizer_path=infer_args.model.phrase_encoder_path,
        retriever_data_file=infer_args.infer.data_file,
        retriever_vector_store_path=infer_args.infer.vector_store_path,
        retriever_save_vector_store_path=infer_args.infer.save_vector_store_path,
    )

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        outputs = infer(
            model,
            phrase_sampler,
            tokenizer,
            retriever,
            queries=[query],
            doc_top_k=infer_args.infer.doc_top_k,
            do_sample=infer_args.infer.do_sample,
            max_new_tokens=infer_args.infer.max_new_tokens,
            temperature=infer_args.infer.temperature,
            top_k=infer_args.infer.top_k,
        )
        print("Assistant: ", end="", flush=True)
        print(outputs[0]["decoded_sentence"])
