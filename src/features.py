from __future__ import annotations

import re
from typing import List

import numpy as np
import pandas as pd


def tokenize_simple(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", str(text).lower())


def word_overlap_ratio(text_a: str, text_b: str) -> float:
    tokens_a = set(tokenize_simple(text_a))
    tokens_b = set(tokenize_simple(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))


def option_length_features(option_df: pd.DataFrame) -> pd.DataFrame:
    frame = option_df.copy()
    frame["article_len"] = frame["article"].fillna("").astype(str).str.len()
    frame["question_len"] = frame["question"].fillna("").astype(str).str.len()
    frame["option_len"] = frame["option_text"].fillna("").astype(str).str.len()
    frame["question_option_overlap"] = [
        word_overlap_ratio(question, option)
        for question, option in zip(frame["question"], frame["option_text"])
    ]
    frame["article_option_overlap"] = [
        word_overlap_ratio(article, option)
        for article, option in zip(frame["article"], frame["option_text"])
    ]
    return frame


def numeric_feature_matrix(option_df: pd.DataFrame) -> np.ndarray:
    frame = option_length_features(option_df)
    columns = [
        "article_len",
        "question_len",
        "option_len",
        "question_option_overlap",
        "article_option_overlap",
    ]
    return frame[columns].astype(float).to_numpy()
