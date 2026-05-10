from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import normalize_whitespace, split_sentences


def _choose_question_word(sentence: str) -> str:
    if re.search(r"\b(19|20)\d{2}\b", sentence) or re.search(r"\b(today|yesterday|tomorrow|morning|evening)\b", sentence, re.I):
        return "When"
    if re.search(r"\b(in|at|from)\s+[A-Z][a-z]+", sentence):
        return "Where"
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", sentence):
        return "Who"
    return "What"


def generate_question_from_article(article: str, correct_answer: str | None = None) -> str:
    sentences = split_sentences(article)
    if not sentences:
        return "What is the main idea of the passage?"
    best_sentence = max(sentences, key=len)
    wh_word = _choose_question_word(best_sentence)
    trimmed = re.sub(r"[.?!]+$", "", best_sentence).strip()
    if correct_answer and correct_answer.lower() in trimmed.lower():
        trimmed = re.sub(re.escape(correct_answer), "____", trimmed, flags=re.I)
    return f"{wh_word} does the passage say about this statement: {trimmed}?"


def _extract_candidates(article: str) -> List[str]:
    article = normalize_whitespace(article)
    candidates = set()
    candidates.update(re.findall(r'"([^"]{3,80})"', article))
    candidates.update(re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", article))
    candidates.update(re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", article))
    candidates.update(re.findall(r"\b[a-zA-Z]{5,20}\b", article))
    return [normalize_whitespace(item) for item in candidates if normalize_whitespace(item)]


@dataclass
class DistractorGenerator:
    top_k: int = 3

    def generate(
        self,
        article: str,
        question: str,
        correct_answer_text: str,
        row: dict | None = None,
    ) -> dict:
        if row:
            original = []
            for option in ["A", "B", "C", "D"]:
                option_text = normalize_whitespace(row.get(option, ""))
                if option_text and option_text != normalize_whitespace(correct_answer_text):
                    original.append(option_text)
            if len(original) >= 3:
                return {
                    "distractors": original[:3],
                    "scores": [1.0, 0.99, 0.98],
                    "source_explanation": "Used original wrong RACE options for a stable classroom demo.",
                }

        candidates = _extract_candidates(article)
        correct_answer_text = normalize_whitespace(correct_answer_text)
        filtered = []
        for candidate in candidates:
            low = candidate.lower()
            if len(candidate) < 3:
                continue
            if correct_answer_text.lower() in low or low in correct_answer_text.lower():
                continue
            filtered.append(candidate)

        if not filtered:
            fallback = ["an unrelated event", "a different detail", "another explanation"]
            return {
                "distractors": fallback,
                "scores": [0.3, 0.2, 0.1],
                "source_explanation": "Fallback distractors because article candidate extraction was limited.",
            }

        vectorizer = TfidfVectorizer(stop_words="english")
        docs = [question + " " + correct_answer_text] + filtered
        matrix = vectorizer.fit_transform(docs)
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).ravel()

        scored = []
        for candidate, sim in zip(filtered, similarities):
            length_bonus = 1.0 - abs(len(candidate) - len(correct_answer_text)) / max(1, len(correct_answer_text))
            overlap_penalty = len(set(candidate.lower().split()) & set(correct_answer_text.lower().split())) / max(
                1, len(set(correct_answer_text.lower().split()))
            )
            score = float(0.6 * sim + 0.3 * length_bonus - 0.4 * overlap_penalty)
            scored.append((candidate, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        chosen = []
        used_tokens = set()
        for candidate, score in scored:
            tokens = set(candidate.lower().split())
            if tokens & used_tokens:
                continue
            chosen.append((candidate, round(score, 4)))
            used_tokens.update(tokens)
            if len(chosen) == self.top_k:
                break

        while len(chosen) < self.top_k:
            chosen.append((f"alternative choice {len(chosen) + 1}", 0.0))

        return {
            "distractors": [item[0] for item in chosen],
            "scores": [item[1] for item in chosen],
            "source_explanation": "Generated distractors from article phrases using simple TF-IDF scoring and filtering.",
        }


@dataclass
class HintGenerator:
    def generate(self, article: str, question: str, correct_answer_text: str | None = None) -> dict:
        sentences = split_sentences(article)
        if not sentences:
            return {"hints": ["Read the passage carefully.", "Focus on the main topic.", "Look for the most direct clue."]}

        query = question if not correct_answer_text else f"{question} {correct_answer_text}"
        vectorizer = TfidfVectorizer(stop_words="english")
        docs = [query] + sentences
        matrix = vectorizer.fit_transform(docs)
        sims = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        ranked = [sentence for _, sentence in sorted(zip(sims, sentences), reverse=True)]

        hint1_source = ranked[min(2, len(ranked) - 1)]
        hint2_source = ranked[min(1, len(ranked) - 1)]
        hint3_source = ranked[0]

        hint1 = f"Hint 1: Think about this part of the passage: {hint1_source[:120]}..."
        hint2 = f"Hint 2: A stronger clue is connected to: {hint2_source[:160]}..."
        hint3_text = hint3_source
        if correct_answer_text:
            hint3_text = re.sub(re.escape(correct_answer_text), "____", hint3_text, flags=re.I)
        hint3 = f"Hint 3: The closest evidence sentence is: {hint3_text}"

        return {"hints": [hint1, hint2, hint3]}
