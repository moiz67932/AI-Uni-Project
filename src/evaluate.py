from __future__ import annotations

from typing import Iterable, List

import nltk
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from src.utils import save_json


def ensure_nltk_resources() -> None:
    resource_paths = {
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "punkt": "tokenizers/punkt",
    }
    for resource, resource_path in resource_paths.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


def bleu_score_single(reference: str, candidate: str) -> float:
    smoothie = SmoothingFunction().method1
    return float(sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie))


def rouge_scores_single(reference: str, candidate: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "rouge1_f1": float(scores["rouge1"].fmeasure),
        "rougeL_f1": float(scores["rougeL"].fmeasure),
    }


def meteor_single(reference: str, candidate: str) -> float:
    return float(meteor_score([reference.split()], candidate.split()))


def average_text_metrics(references: Iterable[str], candidates: Iterable[str]) -> dict:
    ensure_nltk_resources()
    bleu_scores = []
    rouge1_scores = []
    rougeL_scores = []
    meteor_scores = []
    for reference, candidate in zip(references, candidates):
        bleu_scores.append(bleu_score_single(reference, candidate))
        rouge = rouge_scores_single(reference, candidate)
        rouge1_scores.append(rouge["rouge1_f1"])
        rougeL_scores.append(rouge["rougeL_f1"])
        meteor_scores.append(meteor_single(reference, candidate))
    return {
        "bleu": float(np.mean(bleu_scores)) if bleu_scores else 0.0,
        "rouge1_f1": float(np.mean(rouge1_scores)) if rouge1_scores else 0.0,
        "rougeL_f1": float(np.mean(rougeL_scores)) if rougeL_scores else 0.0,
        "meteor": float(np.mean(meteor_scores)) if meteor_scores else 0.0,
        "num_samples": len(bleu_scores),
    }


def save_final_metrics(path, metrics: dict) -> None:
    save_json(path, metrics)
