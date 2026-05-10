from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from src.config import Config
from src.features import numeric_feature_matrix
from src.model_b import DistractorGenerator, HintGenerator, generate_question_from_article
from src.utils import append_inference_log, normalize_whitespace


def _build_option_frame(article: str, question: str, options: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for label in ["A", "B", "C", "D"]:
        option_text = normalize_whitespace(options.get(label, ""))
        rows.append(
            {
                "question_id": 1,
                "article": normalize_whitespace(article),
                "question": normalize_whitespace(question),
                "option_label": label,
                "option_text": option_text,
                "correct_answer": "",
                "label": 0,
                "combined_text": " ".join(
                    [
                        normalize_whitespace(article),
                        normalize_whitespace(article),
                        normalize_whitespace(question),
                        option_text,
                    ]
                ).strip(),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class EnsembleVerifier:
    members: Dict[str, object]
    weights: Dict[str, float]

    def predict_option_probabilities(self, option_df: pd.DataFrame) -> np.ndarray:
        scores = [
            self.weights[name] * _predict_model_probabilities(model, option_df)
            for name, model in self.members.items()
        ]
        return np.sum(scores, axis=0)


def _predict_model_probabilities(model, option_df: pd.DataFrame) -> np.ndarray:
    if isinstance(model, dict) and model.get("model_type") == "tfidf_numeric_logistic_regression":
        X_text = model["vectorizer"].transform(option_df["combined_text"])
        X_numeric = model["scaler"].transform(numeric_feature_matrix(option_df))
        from scipy.sparse import csr_matrix, hstack

        X = hstack([X_text, csr_matrix(X_numeric)], format="csr")
        return model["model"].predict_proba(X)[:, 1]
    if isinstance(model, dict) and "model" in model:
        return model["model"].predict_proba(
            model["scaler"].transform(numeric_feature_matrix(option_df))
        )[:, 1]
    return model.predict_proba(option_df["combined_text"])[:, 1]


def load_model_a(config: Config, model_name: str = "ensemble"):
    model_paths = config.model_a_paths()
    if model_name == "ensemble":
        if model_paths["ensemble"].exists():
            ensemble_config = joblib.load(model_paths["ensemble"])
            required = ensemble_config.get("members", ["count_lr", "tfidf_lr", "tfidf_svm"])
            weights = ensemble_config.get("weights") or {name: 1.0 / len(required) for name in required}
        else:
            required = ["count_lr", "tfidf_lr", "tfidf_svm"]
            weights = {name: 1.0 / len(required) for name in required}
        missing = [name for name in required if not model_paths[name].exists()]
        if missing:
            raise FileNotFoundError("Model files not found. Please run training first.")
        members = {name: joblib.load(model_paths[name]) for name in required}
        return EnsembleVerifier(
            members=members,
            weights=weights,
        )

    if model_name not in model_paths or not model_paths[model_name].exists():
        raise FileNotFoundError("Model files not found. Please run training first.")
    return joblib.load(model_paths[model_name])


def predict_correct_option(
    article: str,
    question: str,
    options: Dict[str, str],
    config: Config | None = None,
    model_name: str = "ensemble",
) -> dict:
    config = config or Config()
    model = load_model_a(config, model_name=model_name)
    option_df = _build_option_frame(article, question, options)
    start_time = time.time()

    if isinstance(model, EnsembleVerifier):
        probabilities = model.predict_option_probabilities(option_df)
    else:
        probabilities = _predict_model_probabilities(model, option_df)

    latency = time.time() - start_time
    best_index = int(np.argmax(probabilities))
    predicted_option = option_df.iloc[best_index]["option_label"]
    return {
        "predicted_option": predicted_option,
        "confidence": float(probabilities[best_index]),
        "option_probabilities": {
            label: float(score) for label, score in zip(option_df["option_label"], probabilities)
        },
        "latency_seconds": round(latency, 4),
    }


def verify_answer(
    article: str,
    question: str,
    options: Dict[str, str],
    selected_option: str,
    config: Config | None = None,
    model_name: str = "ensemble",
) -> dict:
    prediction = predict_correct_option(article, question, options, config=config, model_name=model_name)
    prediction["selected_option"] = selected_option
    prediction["is_correct"] = prediction["predicted_option"] == selected_option
    return prediction


def generate_question(article: str, row: dict | None = None) -> str:
    if row and row.get("question"):
        return normalize_whitespace(row["question"])
    return generate_question_from_article(article)


def generate_distractors(article: str, question: str, correct_answer_text: str, row: dict | None = None) -> dict:
    generator = DistractorGenerator()
    return generator.generate(article, question, correct_answer_text, row=row)


def generate_hints(article: str, question: str, correct_answer_text: str | None = None) -> dict:
    generator = HintGenerator()
    return generator.generate(article, question, correct_answer_text)


def run_full_pipeline(article: str, row: dict | None = None, config: Config | None = None, model_name: str = "ensemble") -> dict:
    config = config or Config()
    article = normalize_whitespace(article)
    question = generate_question(article, row=row)

    if row:
        correct_option = normalize_whitespace(row.get("answer", "A")).upper()
        options = {label: normalize_whitespace(row.get(label, "")) for label in ["A", "B", "C", "D"]}
        correct_answer_text = options.get(correct_option, "")
    else:
        options = {
            "A": "Choice A",
            "B": "Choice B",
            "C": "Choice C",
            "D": "Choice D",
        }
        correct_option = ""
        correct_answer_text = ""

    prediction = None
    try:
        prediction = predict_correct_option(article, question, options, config=config, model_name=model_name)
    except Exception as exc:
        prediction = {"error": str(exc), "latency_seconds": 0.0}

    distractors = generate_distractors(article, question, correct_answer_text, row=row)
    hints = generate_hints(article, question, correct_answer_text)
    result = {
        "article": article,
        "question": question,
        "options": options,
        "correct_option": correct_option,
        "prediction": prediction,
        "distractors": distractors,
        "hints": hints["hints"],
    }
    append_inference_log(
        config.INFERENCE_LOG_FILE,
        {
            "question": question,
            "predicted_option": prediction.get("predicted_option", ""),
            "confidence": prediction.get("confidence", ""),
            "latency_seconds": prediction.get("latency_seconds", 0.0),
            "model_name": model_name,
        },
    )
    return result
