from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import pandas as pd


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_whitespace(text: Any) -> str:
    if pd.isna(text):
        return ""
    text = str(text).replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [item.strip() for item in sentences if item.strip()]


def ensure_list(values: Iterable[Any]) -> List[Any]:
    return list(values)


def append_inference_log(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", index=False, header=False)
    else:
        df.to_csv(path, index=False)
