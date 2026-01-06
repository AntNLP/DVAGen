import copy
import re

import numpy as np
import torch
from nltk import sent_tokenize
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .metric import BaseMetric


AUTOAIS_MODEL = "google/t5_xxl_true_nli_mixture"


def _format_document(doc):
    """Format document for AutoAIS."""
    if "sent" in doc:
        return f"Title: {doc['title']}\n{doc['sent']}"
    else:
        return f"Title: {doc['title']}\n{doc['text']}"


def _run_nli_autoais(passage, claim, model, tokenizer):
    """Run inference for assessing AIS between a premise and hypothesis.

    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py.
    """
    input_text = f"premise: {passage} hypothesis: {claim}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(
        model.device
    )
    with torch.inference_mode():
        outputs = model.generate(input_ids, max_new_tokens=10)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference


def _get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    max_memory = f"{free_in_GB - 6}GB"
    n_gpus = torch.cuda.device_count()
    max_memory = dict.fromkeys(range(n_gpus), max_memory)
    return max_memory


def _remove_citations(sent):
    return (
        re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent))
        .replace(" |", "")
        .replace("]", "")
    )


class CitationScoreMetric(BaseMetric):
    def __init__(
        self,
        predictions: list[str],
        references: list[dict],
        model_name_path: str,
    ) -> None:
        self.predictions = predictions
        self.references = references
        self.length = len(predictions)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            AUTOAIS_MODEL,
            torch_dtype=torch.bfloat16,
            max_memory=_get_max_memory(),
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            AUTOAIS_MODEL, use_fast=False
        )

    def compute(self) -> dict:
        ais_scores = []
        ais_scores_prec = []

        sent_total = 0
        sent_mcite = 0
        sent_mcite_support = 0
        sent_mcite_overcite = 0
        autoais_log = []

        for idx in tqdm(range(len(self.predictions))):
            # Get passage by using citation
            joint_passage = "\n".join(
                [
                    _format_document(self.references[idx]["docs"][psgs_id])
                    for psgs_id in self.references[idx]["docs_id"]
                ]
            )
            # Get sentences by using NLTK
            sents = sent_tokenize(self.references[idx]["output"])
            if len(sents) == 0:
                continue

            target_sents = [_remove_citations(sent).strip() for sent in sents]

            entail = 0
            entail_prec = 0
            total_citations = 0
            for sent_id, sent in enumerate(sents):
                target_sent = target_sents[
                    sent_id
                ]  # Citation removed and (if opted for) decontextualized
                joint_entail = -1  # Undecided

                # Find references
                ref = [
                    int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)
                ]  # In text citation id starts from 1
                if len(ref) == 0:
                    # No citations
                    joint_entail = 0
                elif any(
                    [
                        ref_id >= len(self.references[idx]["docs"])
                        for ref_id in ref
                    ]
                ):
                    # Citations out of range
                    joint_entail = 0
                else:
                    total_citations += len(ref)
                    joint_passage = "\n".join(
                        [
                            _format_document(
                                self.references[idx]["docs"][psgs_id]
                            )
                            for psgs_id in ref
                        ]
                    )

                # If not directly rejected by citation format error, calculate the recall score
                if joint_entail == -1:
                    joint_entail = _run_nli_autoais(
                        joint_passage, target_sent, self.model, self.tokenizer
                    )
                    autoais_log.append(
                        {
                            "question": self.references[idx]["question"],
                            "output": self.predictions[idx],
                            "claim": sent,
                            "passage": [joint_passage],
                            "model_type": "NLI",
                            "model_output": joint_entail,
                        }
                    )

                entail += joint_entail
                if len(ref) > 1:
                    sent_mcite += 1

                # calculate the precision score if applicable
                if joint_entail and len(ref) > 1:
                    sent_mcite_support += 1
                    # Precision check: did the model cite any unnecessary documents?
                    for psgs_id in ref:
                        # condition A
                        passage = _format_document(
                            self.references[idx]["docs"][psgs_id]
                        )
                        nli_result = _run_nli_autoais(
                            passage, target_sent, self.model, self.tokenizer
                        )

                        # condition B
                        if not nli_result:
                            subset_exclude = copy.deepcopy(ref)
                            subset_exclude.remove(psgs_id)
                            passage = "\n".join(
                                [
                                    _format_document(
                                        self.references[idx]["docs"][pid]
                                    )
                                    for pid in subset_exclude
                                ]
                            )
                            nli_result = _run_nli_autoais(
                                passage, target_sent, self.model, self.tokenizer
                            )
                            if nli_result:  # psgs_id is not necessary
                                flag = 0
                                sent_mcite_overcite += 1
                            else:
                                entail_prec += 1
                        else:
                            entail_prec += 1
                else:
                    entail_prec += joint_entail

            sent_total += len(sents)
            ais_scores.append(entail / len(sents))
            ais_scores_prec.append(
                entail_prec / total_citations if total_citations > 0 else 0
            )  # len(sents))

        if sent_mcite > 0 and sent_mcite_support > 0:
            print(
                f"Among all sentences, "
                f"{100 * sent_mcite / sent_total:.2f} have multiple citations, "
                f"among which {100 * sent_mcite_support / sent_mcite:.2f} are supported by the joint set, "
                f"among which {100 * sent_mcite_overcite / sent_mcite_support:.2f} overcite."
            )

        return {
            "citation_rec": 100 * np.mean(ais_scores),
            "citation_prec": 100 * np.mean(ais_scores_prec),
        }
