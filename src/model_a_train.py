from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from src.checkpointing import CheckpointManager, safe_save
from src.config import Config
from src.features import numeric_feature_matrix
from src.utils import save_json


def build_model_pipelines(config: Config) -> Dict[str, Pipeline]:
    count_lr = Pipeline(
        [
            (
                "vectorizer",
                CountVectorizer(
                    binary=True,
                    max_features=config.count_max_features,
                    stop_words="english",
                    ngram_range=config.ngram_range,
                    min_df=config.min_df,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=config.random_seed,
                ),
            ),
        ]
    )
    tfidf_lr = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    max_features=config.tfidf_max_features,
                    stop_words="english",
                    sublinear_tf=True,
                    ngram_range=config.ngram_range,
                    min_df=config.min_df,
                    max_df=config.max_df,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=config.random_seed,
                ),
            ),
        ]
    )
    tfidf_svm = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    max_features=config.tfidf_max_features,
                    stop_words="english",
                    sublinear_tf=True,
                    ngram_range=config.ngram_range,
                    min_df=config.min_df,
                    max_df=config.max_df,
                ),
            ),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=LinearSVC(class_weight="balanced", random_state=config.random_seed),
                    cv=3,
                ),
            ),
        ]
    )
    return {
        "count_lr": count_lr,
        "tfidf_lr": tfidf_lr,
        "tfidf_svm": tfidf_svm,
    }


def train_or_load_pipeline(
    name: str,
    pipeline: Pipeline,
    path: Path,
    train_texts: pd.Series,
    train_labels: pd.Series,
) -> Pipeline:
    if path.exists():
        try:
            return joblib.load(path)
        except Exception as exc:
            corrupt_path = path.with_suffix(path.suffix + ".corrupt")
            path.replace(corrupt_path)
            print(f"Warning: {path.name} was corrupted and renamed to {corrupt_path.name}. Retraining. {exc}")

    pipeline.fit(train_texts, train_labels)
    safe_save(path, pipeline)
    return pipeline


def train_or_load_random_forest(
    config: Config,
    train_opt: pd.DataFrame,
    path: Path,
) -> dict:
    if path.exists():
        try:
            return joblib.load(path)
        except Exception as exc:
            corrupt_path = path.with_suffix(path.suffix + ".corrupt")
            path.replace(corrupt_path)
            print(f"Warning: {path.name} was corrupted and renamed to {corrupt_path.name}. Retraining. {exc}")

    X_train = numeric_feature_matrix(train_opt)
    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=config.random_seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(X_train_scaled, train_opt["label"])
    package = {"scaler": scaler, "model": model}
    safe_save(path, package)
    return package


def _safe_predict_proba(model, texts: pd.Series) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(texts)
        scores = np.asarray(scores)
        return 1.0 / (1.0 + np.exp(-scores))
    preds = model.predict(texts)
    return np.asarray(preds, dtype=float)


def evaluate_option_level_from_scores(
    option_df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_true = option_df["label"].to_numpy()
    y_pred = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision_pos, recall_pos, f1_pos, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], zero_division=0
    )
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_class_precision": float(precision_pos[0]),
        "positive_class_recall": float(recall_pos[0]),
        "positive_class_f1": float(f1_pos[0]),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "failed_positive_detection": bool(f1_pos[0] == 0.0),
    }


def evaluate_question_level(option_df: pd.DataFrame, probabilities: np.ndarray) -> dict:
    scored = option_df[["question_id", "option_label", "correct_answer"]].copy()
    scored["score"] = probabilities
    best = scored.sort_values(["question_id", "score"], ascending=[True, False]).groupby("question_id").head(1)
    accuracy = (best["option_label"] == best["correct_answer"]).mean()
    return {
        "question_level_accuracy": float(accuracy),
        "num_questions": int(best["question_id"].nunique()),
    }


def plot_confusion_matrix(cm: list[list[int]], path: Path) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.title("Model A Confusion Matrix")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def train_and_evaluate_model_a(
    config: Config,
    train_opt: pd.DataFrame,
    val_opt: pd.DataFrame,
    test_opt: pd.DataFrame,
    quick: bool = False,
) -> Tuple[dict, dict]:
    model_paths = config.model_a_paths()
    checkpoint_manager = CheckpointManager(config.MODEL_A_CHECKPOINT_DIR)
    if quick:
        train_opt = train_opt.head(min(len(train_opt), config.quick_train_rows))
        val_opt = val_opt.head(min(len(val_opt), max(200, config.quick_train_rows // 5)))
        test_opt = test_opt.head(min(len(test_opt), max(200, config.quick_train_rows // 5)))

    results = {"models": {}, "notes": []}
    train_texts = train_opt["combined_text"]
    train_labels = train_opt["label"]
    checkpoint_manager.save("option_train.joblib", train_opt, description="Training option-level dataframe")
    checkpoint_manager.save("option_val.joblib", val_opt, description="Validation option-level dataframe")
    checkpoint_manager.save("option_test.joblib", test_opt, description="Test option-level dataframe")
    checkpoint_manager.save("y_train.joblib", train_labels, description="Training labels")
    checkpoint_manager.save("y_val.joblib", val_opt["label"], description="Validation labels")
    checkpoint_manager.save("y_test.joblib", test_opt["label"], description="Test labels")

    pipelines = build_model_pipelines(config)
    trained_models = {}
    for name, pipeline in pipelines.items():
        trained = train_or_load_pipeline(name, pipeline, model_paths[name], train_texts, train_labels)
        trained_models[name] = trained
        vectorizer = trained.named_steps["vectorizer"]
        checkpoint_name = "vectorizer_count.joblib" if name == "count_lr" else "vectorizer_tfidf.joblib"
        if not checkpoint_manager.exists(checkpoint_name):
            checkpoint_manager.save(checkpoint_name, vectorizer, description=f"{name} vectorizer")
        probabilities = _safe_predict_proba(trained, test_opt["combined_text"])
        option_metrics = evaluate_option_level_from_scores(test_opt, probabilities)
        question_metrics = evaluate_question_level(test_opt, probabilities)
        results["models"][name] = {
            "option_level": option_metrics,
            "question_level": question_metrics,
        }
        if option_metrics["failed_positive_detection"]:
            results["notes"].append(f"{name} predicts the positive class poorly and should be treated as weak.")

    rf_package = train_or_load_random_forest(config, train_opt, model_paths["random_forest"])
    X_test_rf = numeric_feature_matrix(test_opt)
    X_test_rf = rf_package["scaler"].transform(X_test_rf)
    rf_probabilities = rf_package["model"].predict_proba(X_test_rf)[:, 1]
    results["models"]["random_forest"] = {
        "option_level": evaluate_option_level_from_scores(test_opt, rf_probabilities),
        "question_level": evaluate_question_level(test_opt, rf_probabilities),
    }
    trained_models["random_forest"] = rf_package

    ensemble_probabilities = np.mean(
        [
            _safe_predict_proba(trained_models["count_lr"], test_opt["combined_text"]),
            _safe_predict_proba(trained_models["tfidf_lr"], test_opt["combined_text"]),
            _safe_predict_proba(trained_models["tfidf_svm"], test_opt["combined_text"]),
        ],
        axis=0,
    )
    ensemble_metrics = {
        "option_level": evaluate_option_level_from_scores(test_opt, ensemble_probabilities),
        "question_level": evaluate_question_level(test_opt, ensemble_probabilities),
    }
    results["models"]["ensemble"] = ensemble_metrics
    safe_save(
        model_paths["ensemble"],
        {
            "type": "simple_average",
            "members": ["count_lr_pipeline.joblib", "tfidf_lr_pipeline.joblib", "tfidf_svm_pipeline.joblib"],
        },
    )

    sample_predictions = test_opt[
        ["question_id", "option_label", "correct_answer", "label", "combined_text"]
    ].copy()
    sample_predictions["ensemble_probability"] = ensemble_probabilities
    sample_predictions["count_lr_probability"] = _safe_predict_proba(trained_models["count_lr"], test_opt["combined_text"])
    sample_predictions["tfidf_lr_probability"] = _safe_predict_proba(trained_models["tfidf_lr"], test_opt["combined_text"])
    sample_predictions["tfidf_svm_probability"] = _safe_predict_proba(trained_models["tfidf_svm"], test_opt["combined_text"])
    sample_predictions.to_csv(config.SAMPLE_PREDICTIONS_FILE, index=False)

    plot_confusion_matrix(
        results["models"]["tfidf_svm"]["option_level"]["confusion_matrix"],
        config.CONFUSION_MATRIX_FIG,
    )
    save_json(config.MODEL_A_RESULTS_FILE, results)
    return results, trained_models


def train_unsupervised_component(
    config: Config,
    train_opt: pd.DataFrame,
    quick: bool = False,
) -> dict:
    sample_size = min(len(train_opt), 2000 if quick else config.kmeans_sample_size)
    sample_df = train_opt.sample(n=sample_size, random_state=config.random_seed) if len(train_opt) > sample_size else train_opt.copy()

    vectorizer = TfidfVectorizer(
        max_features=min(8000, config.tfidf_max_features),
        stop_words="english",
        sublinear_tf=True,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
    )
    X_sample = vectorizer.fit_transform(sample_df["combined_text"])
    n_clusters = 4
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=config.random_seed, batch_size=256)
    clusters = model.fit_predict(X_sample)
    sample_df = sample_df.copy()
    sample_df["cluster"] = clusters

    majority_map = {}
    purity_scores = []
    distribution = Counter(clusters)
    for cluster_id, cluster_frame in sample_df.groupby("cluster"):
        majority_label = int(cluster_frame["label"].mode().iloc[0])
        majority_map[int(cluster_id)] = majority_label
        purity = float((cluster_frame["label"] == majority_label).mean())
        purity_scores.append(purity)

    silhouette = None
    if X_sample.shape[0] > n_clusters:
        try:
            silhouette = float(silhouette_score(X_sample, clusters))
        except Exception:
            silhouette = None

    result = {
        "sample_size": int(X_sample.shape[0]),
        "n_clusters": n_clusters,
        "cluster_distribution": {str(k): int(v) for k, v in distribution.items()},
        "majority_label_per_cluster": {str(k): int(v) for k, v in majority_map.items()},
        "cluster_purity_mean": float(np.mean(purity_scores)) if purity_scores else 0.0,
        "silhouette_score": silhouette,
    }

    package = {"vectorizer": vectorizer, "model": model, "result": result}
    safe_save(config.model_a_paths()["kmeans"], package)
    save_json(config.UNSUPERVISED_RESULTS_FILE, result)
    return result
