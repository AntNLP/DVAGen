import os
import signal
import subprocess
import sys
from pathlib import Path
from venv import logger

from ..configs.parser import InferArgs, parse_args
from ..utils import logging
from .infer import infer, prepare

logger = logging.get_logger(__name__)


def chat(infer_args: InferArgs):
    if infer_args.infer.mode == "CLI":
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
        while True:
            try:
                query = input("\nUser: ")
            except UnicodeDecodeError:
                print(
                    "Detected decoding error at the inputs, please set the terminal encoding to utf-8."
                )
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
    elif infer_args.infer.mode == "WebUI":
        dva_root = str(Path(__file__).resolve().parent.parent.parent.parent)
        web_root = str(Path(__file__).resolve().parent.parent / "web")
        config_path: str = str(parse_args(sys.argv).pop("config_path"))
        env = os.environ.copy()
        env.update(
            {
                "config_path": config_path,
                "dvagen_root": dva_root,
                "web_root": web_root,
            }
        )
        logger.info("Starting DVAGen Web Server...")
        proc = subprocess.Popen(
            [
                sys.executable,
                "manage.py",
                "runserver",
                "--insecure",
                f"0.0.0.0:{infer_args.infer.port}",
            ],
            cwd=web_root,
            env=env,
        )

        def _graceful(signum, frame):
            logger.info("Shutting down DVAGen Web Server...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            sys.exit(0)

        signal.signal(signal.SIGINT, _graceful)
        signal.signal(signal.SIGTERM, _graceful)

        proc.wait()
    else:
        raise NotImplementedError
