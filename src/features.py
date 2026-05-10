from __future__ import annotations

from functools import lru_cache
import re
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


FEATURE_COLUMNS = [
    "article_len",
    "question_len",
    "option_len",
    "option_word_count",
    "question_word_count",
    "question_option_overlap",
    "article_option_overlap",
    "option_in_article",
    "option_lower_in_article_lower",
    "max_sentence_option_overlap",
    "max_sentence_question_option_overlap",
    "max_sentence_tfidf_similarity",
    "option_is_number",
    "option_is_year",
    "option_has_capitalized_word",
]


def tokenize_simple(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", str(text).lower())


def sentence_split_simple(text: str) -> List[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text)) if part.strip()]
    return sentences or [str(text)]


def word_overlap_ratio(text_a: str, text_b: str) -> float:
    tokens_a = set(tokenize_simple(text_a))
    tokens_b = set(tokenize_simple(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))


def _is_number(text: str) -> bool:
    return bool(re.fullmatch(r"\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+)%?\s*", str(text)))


def _is_year(text: str) -> bool:
    return bool(re.fullmatch(r"\s*(?:1[5-9]\d{2}|20\d{2}|21\d{2})\s*", str(text)))


@lru_cache(maxsize=20000)
def _sentence_features(article: str, question: str, option: str) -> tuple[float, float, float]:
    sentences = sentence_split_simple(article)
    max_option_overlap = max((word_overlap_ratio(sentence, option) for sentence in sentences), default=0.0)
    question_option = f"{question} {option}".strip()
    max_question_option_overlap = max(
        (word_overlap_ratio(sentence, question_option) for sentence in sentences),
        default=0.0,
    )

    max_tfidf_similarity = 0.0
    if question_option and any(tokenize_simple(sentence) for sentence in sentences):
        try:
            vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
            matrix = vectorizer.fit_transform(sentences + [question_option])
            if matrix.shape[1] > 0 and matrix.shape[0] > 1:
                similarities = cosine_similarity(matrix[:-1], matrix[-1])
                max_tfidf_similarity = float(np.max(similarities))
        except ValueError:
            max_tfidf_similarity = 0.0

    return float(max_option_overlap), float(max_question_option_overlap), max_tfidf_similarity


def option_length_features(option_df: pd.DataFrame) -> pd.DataFrame:
    frame = option_df.copy()
    article = frame["article"].fillna("").astype(str)
    question = frame["question"].fillna("").astype(str)
    option = frame["option_text"].fillna("").astype(str)

    frame["article_len"] = article.str.len()
    frame["question_len"] = question.str.len()
    frame["option_len"] = option.str.len()
    frame["option_word_count"] = option.apply(lambda text: len(tokenize_simple(text)))
    frame["question_word_count"] = question.apply(lambda text: len(tokenize_simple(text)))
    frame["question_option_overlap"] = [
        word_overlap_ratio(question, option)
        for question, option in zip(question, option)
    ]
    frame["article_option_overlap"] = [
        word_overlap_ratio(article, option)
        for article, option in zip(article, option)
    ]
    frame["option_in_article"] = [int(opt in art) for art, opt in zip(article, option)]
    frame["option_lower_in_article_lower"] = [
        int(opt.lower() in art.lower()) for art, opt in zip(article, option)
    ]

    sentence_values = [
        _sentence_features(art, ques, opt)
        for art, ques, opt in zip(article, question, option)
    ]
    frame["max_sentence_option_overlap"] = [value[0] for value in sentence_values]
    frame["max_sentence_question_option_overlap"] = [value[1] for value in sentence_values]
    frame["max_sentence_tfidf_similarity"] = [value[2] for value in sentence_values]
    frame["option_is_number"] = option.apply(lambda text: int(_is_number(text)))
    frame["option_is_year"] = option.apply(lambda text: int(_is_year(text)))
    frame["option_has_capitalized_word"] = option.apply(
        lambda text: int(bool(re.search(r"\b[A-Z][a-z]+\b", str(text))))
    )
    return frame


def numeric_feature_matrix(option_df: pd.DataFrame) -> np.ndarray:
    frame = option_length_features(option_df)
    return frame[FEATURE_COLUMNS].astype(float).to_numpy()
