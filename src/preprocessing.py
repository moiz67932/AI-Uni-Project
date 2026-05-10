from __future__ import annotations

import shutil
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.config import Config
from src.utils import save_json
from src.utils import normalize_whitespace


def select_single_source_csv(config: Config) -> Path:
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if config.DATA_FILE.exists():
        return config.DATA_FILE

    csv_files = [path for path in config.RAW_DATA_DIR.rglob("*.csv") if path.name != "race_single.csv"]
    if not csv_files:
        raise FileNotFoundError(
            "Dataset not found. Please place a CSV in data/raw or run the Kaggle download step."
        )

    preferred = [path for path in csv_files if path.name.lower() == "train.csv"]
    candidates = preferred or csv_files
    selected = None
    best_rows = -1
    for path in candidates:
        try:
            df = pd.read_csv(path)
            if not set(config.required_columns).issubset(df.columns):
                continue
            if len(df) > best_rows:
                best_rows = len(df)
                selected = path
        except Exception:
            continue
    if selected is None:
        raise ValueError("No CSV under data/raw matched the required RACE columns.")

    shutil.copy2(selected, config.DATA_FILE)
    return config.DATA_FILE


def load_raw_dataset(config: Config) -> pd.DataFrame:
    source_path = select_single_source_csv(config)
    return pd.read_csv(source_path)


def clean_dataset(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    df = df.copy()
    before_missing = df[config.required_columns].isnull().sum().to_dict() if set(config.required_columns).issubset(df.columns) else {}
    df = df[[col for col in config.required_columns if col in df.columns]].copy()
    df = df.dropna(subset=config.required_columns)

    for column in config.required_columns:
        df[column] = df[column].apply(normalize_whitespace)

    df["answer"] = df["answer"].str.upper()
    df = df[df["answer"].isin(config.valid_answers)]
    df = df.drop_duplicates().reset_index(drop=True)
    df["question_id"] = range(1, len(df) + 1)
    df.attrs["missing_before_cleaning"] = before_missing
    df.attrs["missing_after_cleaning"] = df[config.required_columns].isnull().sum().to_dict()
    return df


def create_80_10_10_split(
    df: pd.DataFrame, config: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - config.train_size),
        random_state=config.random_seed,
        stratify=df["answer"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=config.random_seed,
        stratify=temp_df["answer"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_option_level(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for option_label in ["A", "B", "C", "D"]:
            option_text = normalize_whitespace(row[option_label])
            combined_text = " ".join(
                [
                    normalize_whitespace(row["article"]),
                    normalize_whitespace(row["article"]),
                    normalize_whitespace(row["question"]),
                    option_text,
                ]
            ).strip()
            rows.append(
                {
                    "question_id": row["question_id"],
                    "article": row["article"],
                    "question": row["question"],
                    "option_label": option_label,
                    "option_text": option_text,
                    "correct_answer": row["answer"],
                    "label": int(option_label == row["answer"]),
                    "combined_text": combined_text,
                }
            )
    option_df = pd.DataFrame(rows)
    return option_df


def save_processed_files(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_opt: pd.DataFrame,
    val_opt: pd.DataFrame,
    test_opt: pd.DataFrame,
    config: Config,
) -> None:
    train_df.to_csv(config.TRAIN_CSV, index=False)
    val_df.to_csv(config.VAL_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)
    train_opt.to_csv(config.TRAIN_OPTION_CSV, index=False)
    val_opt.to_csv(config.VAL_OPTION_CSV, index=False)
    test_opt.to_csv(config.TEST_OPTION_CSV, index=False)


def prepare_processed_data(config: Config):
    raw_df = load_raw_dataset(config)
    clean_df = clean_dataset(raw_df, config)
    train_df, val_df, test_df = create_80_10_10_split(clean_df, config)
    train_opt = make_option_level(train_df)
    val_opt = make_option_level(val_df)
    test_opt = make_option_level(test_df)
    save_processed_files(train_df, val_df, test_df, train_opt, val_opt, test_opt, config)
    return {
        "raw_df": raw_df,
        "clean_df": clean_df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_opt": train_opt,
        "val_opt": val_opt,
        "test_opt": test_opt,
    }


def run_eda(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_opt: pd.DataFrame,
    val_opt: pd.DataFrame,
    test_opt: pd.DataFrame,
    config: Config,
) -> dict:
    summary = {
        "raw_row_count": int(len(raw_df)),
        "clean_row_count": int(len(clean_df)),
        "missing_before_cleaning": raw_df[config.required_columns].isnull().sum().to_dict(),
        "missing_after_cleaning": clean_df[config.required_columns].isnull().sum().to_dict(),
        "answer_distribution": clean_df["answer"].value_counts().sort_index().to_dict(),
        "article_length_stats": clean_df["article"].astype(str).str.len().describe().to_dict(),
        "question_length_stats": clean_df["question"].astype(str).str.len().describe().to_dict(),
        "option_length_stats": pd.concat(
            [clean_df[col].astype(str).str.len() for col in ["A", "B", "C", "D"]], ignore_index=True
        ).describe().to_dict(),
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "option_level_sizes": {
            "train": int(len(train_opt)),
            "val": int(len(val_opt)),
            "test": int(len(test_opt)),
        },
        "option_level_class_balance": {
            "train_positive_rate": float(train_opt["label"].mean()),
            "val_positive_rate": float(val_opt["label"].mean()),
            "test_positive_rate": float(test_opt["label"].mean()),
        },
    }

    plt.figure(figsize=(6, 4))
    sns.countplot(x="answer", data=clean_df, order=["A", "B", "C", "D"], palette="Blues")
    plt.title("Answer Distribution")
    plt.tight_layout()
    config.ANSWER_DISTRIBUTION_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.ANSWER_DISTRIBUTION_FIG)
    plt.close()

    save_json(config.EDA_SUMMARY_FILE, summary)
    return summary
