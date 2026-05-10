from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Config:
    base_dir: str | None = None
    random_seed: int = 42
    train_size: float = 0.80
    val_size: float = 0.10
    test_size: float = 0.10
    valid_answers: List[str] = field(default_factory=lambda: ["A", "B", "C", "D"])
    required_columns: List[str] = field(
        default_factory=lambda: ["article", "question", "A", "B", "C", "D", "answer"]
    )
    tfidf_max_features: int = 20000
    count_max_features: int = 20000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    kmeans_sample_size: int = 10000
    quick_train_rows: int = 1000

    def __post_init__(self) -> None:
        default_base = str(Path.cwd())
        chosen_base = self.base_dir or os.getenv("AI_RACE_PROJECT_BASE_DIR", default_base)
        self.BASE_DIR = Path(chosen_base)

        self.DATA_DIR = self.BASE_DIR / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"

        self.CHECKPOINT_DIR = self.BASE_DIR / "checkpoints"
        self.MODEL_A_CHECKPOINT_DIR = self.CHECKPOINT_DIR / "model_a"
        self.MODEL_B_CHECKPOINT_DIR = self.CHECKPOINT_DIR / "model_b"

        self.MODELS_DIR = self.BASE_DIR / "models"
        self.MODEL_A_MODELS_DIR = self.MODELS_DIR / "model_a" / "traditional"
        self.MODEL_B_MODELS_DIR = self.MODELS_DIR / "model_b" / "traditional"

        self.NOTEBOOKS_DIR = self.BASE_DIR / "notebooks"
        self.SRC_DIR = self.BASE_DIR / "src"
        self.UI_DIR = self.BASE_DIR / "ui"

        self.OUTPUTS_DIR = self.BASE_DIR / "outputs"
        self.RESULTS_DIR = self.OUTPUTS_DIR / "results"
        self.LOGS_DIR = self.OUTPUTS_DIR / "logs"
        self.FIGURES_DIR = self.OUTPUTS_DIR / "figures"

        self.REPORT_DIR = self.BASE_DIR / "report"
        self.SCREENSHOTS_DIR = self.REPORT_DIR / "screenshots"

        self.DATA_FILE = self.RAW_DATA_DIR / "race_single.csv"

        self.TRAIN_CSV = self.PROCESSED_DATA_DIR / "train_80.csv"
        self.VAL_CSV = self.PROCESSED_DATA_DIR / "val_10.csv"
        self.TEST_CSV = self.PROCESSED_DATA_DIR / "test_10.csv"

        self.TRAIN_OPTION_CSV = self.PROCESSED_DATA_DIR / "train_option_level.csv"
        self.VAL_OPTION_CSV = self.PROCESSED_DATA_DIR / "val_option_level.csv"
        self.TEST_OPTION_CSV = self.PROCESSED_DATA_DIR / "test_option_level.csv"

        self.MODEL_A_RESULTS_FILE = self.RESULTS_DIR / "model_a_results.json"
        self.MODEL_B_RESULTS_FILE = self.RESULTS_DIR / "model_b_results.json"
        self.UNSUPERVISED_RESULTS_FILE = self.RESULTS_DIR / "unsupervised_results.json"
        self.FINAL_METRICS_FILE = self.RESULTS_DIR / "final_metrics.json"
        self.EDA_SUMMARY_FILE = self.RESULTS_DIR / "eda_summary.json"
        self.SAMPLE_PREDICTIONS_FILE = self.RESULTS_DIR / "sample_predictions.csv"
        self.INFERENCE_LOG_FILE = self.LOGS_DIR / "inference_logs.csv"

        self.ANSWER_DISTRIBUTION_FIG = self.FIGURES_DIR / "answer_distribution.png"
        self.CONFUSION_MATRIX_FIG = self.FIGURES_DIR / "confusion_matrix_model_a.png"

    def model_a_paths(self) -> Dict[str, Path]:
        return {
            "count_lr": self.MODEL_A_MODELS_DIR / "count_lr_pipeline.joblib",
            "tfidf_lr": self.MODEL_A_MODELS_DIR / "tfidf_lr_pipeline.joblib",
            "tfidf_svm": self.MODEL_A_MODELS_DIR / "tfidf_svm_pipeline.joblib",
            "random_forest": self.MODEL_A_MODELS_DIR / "random_forest_features.joblib",
            "feature_logistic_regression": self.MODEL_A_MODELS_DIR / "feature_logistic_regression.joblib",
            "tfidf_numeric_lr": self.MODEL_A_MODELS_DIR / "tfidf_numeric_lr.joblib",
            "kmeans": self.MODEL_A_MODELS_DIR / "kmeans_model.joblib",
            "ensemble": self.MODEL_A_MODELS_DIR / "ensemble_model.joblib",
        }

    def model_b_paths(self) -> Dict[str, Path]:
        return {
            "distractor_generator": self.MODEL_B_MODELS_DIR / "distractor_generator.joblib",
            "hint_generator": self.MODEL_B_MODELS_DIR / "hint_generator.joblib",
        }


def ensure_project_dirs(config: Config) -> None:
    dirs = [
        config.BASE_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODEL_A_CHECKPOINT_DIR,
        config.MODEL_B_CHECKPOINT_DIR,
        config.MODEL_A_MODELS_DIR,
        config.MODEL_B_MODELS_DIR,
        config.NOTEBOOKS_DIR,
        config.SRC_DIR,
        config.UI_DIR,
        config.RESULTS_DIR,
        config.LOGS_DIR,
        config.FIGURES_DIR,
        config.REPORT_DIR,
        config.SCREENSHOTS_DIR,
    ]
    for path in dirs:
        Path(path).mkdir(parents=True, exist_ok=True)
