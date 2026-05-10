from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.checkpointing import CheckpointManager, safe_save
from src.config import Config, ensure_project_dirs
from src.evaluate import average_text_metrics, save_final_metrics
from src.model_a_train import train_and_evaluate_model_a, train_unsupervised_component
from src.model_b import DistractorGenerator, HintGenerator, generate_question_from_article
from src.preprocessing import prepare_processed_data, run_eda
from src.utils import save_json, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI RACE project runner")
    parser.add_argument("--quick", action="store_true", help="Run a smaller fast demo")
    parser.add_argument("--retrain", action="store_true", help="Retrain models even if files already exist")
    parser.add_argument("--sample-size", type=int, default=10000, help="Sample size for unsupervised component")
    parser.add_argument("--base-dir", type=str, default=None, help="Override project base directory")
    return parser.parse_args()


def maybe_remove_existing_models(config: Config) -> None:
    for path in list(config.model_a_paths().values()) + list(config.model_b_paths().values()):
        if path.exists():
            path.unlink()


def save_split_checkpoints(config: Config, prepared: dict) -> None:
    checkpoint_manager = CheckpointManager(config.MODEL_A_CHECKPOINT_DIR)
    checkpoint_manager.save("df_clean.joblib", prepared["clean_df"], description="Cleaned dataset")
    checkpoint_manager.save("split_train.joblib", prepared["train_df"], description="80 percent train split")
    checkpoint_manager.save("split_val.joblib", prepared["val_df"], description="10 percent validation split")
    checkpoint_manager.save("split_test.joblib", prepared["test_df"], description="10 percent test split")


def save_sparse_feature_checkpoints(config: Config, trained_models: dict, prepared: dict) -> None:
    train_text = prepared["train_opt"]["combined_text"]
    val_text = prepared["val_opt"]["combined_text"]
    test_text = prepared["test_opt"]["combined_text"]

    count_vectorizer = trained_models["count_lr"].named_steps["vectorizer"]
    tfidf_vectorizer = trained_models["tfidf_lr"].named_steps["vectorizer"]

    X_train_count = count_vectorizer.transform(train_text)
    X_val_count = count_vectorizer.transform(val_text)
    X_test_count = count_vectorizer.transform(test_text)

    X_train_tfidf = tfidf_vectorizer.transform(train_text)
    X_val_tfidf = tfidf_vectorizer.transform(val_text)
    X_test_tfidf = tfidf_vectorizer.transform(test_text)

    checkpoint_manager = CheckpointManager(config.MODEL_A_CHECKPOINT_DIR)
    checkpoint_manager.save("vectorizer_count.joblib", count_vectorizer, description="CountVectorizer")
    checkpoint_manager.save("vectorizer_tfidf.joblib", tfidf_vectorizer, description="TF-IDF vectorizer")
    checkpoint_manager.save("X_train_count.joblib", X_train_count, description="Sparse count train matrix")
    checkpoint_manager.save("X_val_count.joblib", X_val_count, description="Sparse count validation matrix")
    checkpoint_manager.save("X_test_count.joblib", X_test_count, description="Sparse count test matrix")
    checkpoint_manager.save("X_train_tfidf.joblib", X_train_tfidf, description="Sparse TF-IDF train matrix")
    checkpoint_manager.save("X_val_tfidf.joblib", X_val_tfidf, description="Sparse TF-IDF validation matrix")
    checkpoint_manager.save("X_test_tfidf.joblib", X_test_tfidf, description="Sparse TF-IDF test matrix")


def evaluate_model_b(config: Config, test_df: pd.DataFrame, quick: bool = False) -> dict:
    sample_df = test_df.head(25 if quick else 100).copy()
    distractor_generator = DistractorGenerator()
    hint_generator = HintGenerator()
    safe_save(config.model_b_paths()["distractor_generator"], distractor_generator)
    safe_save(config.model_b_paths()["hint_generator"], hint_generator)

    generated_questions = []
    reference_questions = []

    generated_distractors = []
    reference_distractors = []

    generated_hints = []
    reference_hint_sentences = []

    samples = []
    for _, row in sample_df.iterrows():
        article = str(row["article"])
        question = str(row["question"])
        correct_option = str(row["answer"]).upper()
        correct_answer = str(row[correct_option])

        gen_question = generate_question_from_article(article, correct_answer=correct_answer)
        generated_questions.append(gen_question)
        reference_questions.append(question)

        distractor_info = distractor_generator.generate(article, question, correct_answer, row=row.to_dict())
        wrong_options = [str(row[label]) for label in ["A", "B", "C", "D"] if label != correct_option]
        generated_distractors.extend(distractor_info["distractors"])
        reference_distractors.extend(wrong_options[: len(distractor_info["distractors"])])

        hints = hint_generator.generate(article, question, correct_answer)["hints"]
        generated_hints.append(hints[-1])
        reference_hint_sentences.append(article.split(".")[0].strip() if "." in article else article[:150])

        samples.append(
            {
                "question_id": row["question_id"],
                "original_question": question,
                "generated_question": gen_question,
                "correct_answer": correct_answer,
                "generated_distractors": distractor_info["distractors"],
                "generated_hints": hints,
            }
        )

    question_metrics = average_text_metrics(reference_questions, generated_questions)
    distractor_metrics = average_text_metrics(reference_distractors, generated_distractors)
    hint_metrics = average_text_metrics(reference_hint_sentences, generated_hints)

    result = {
        "question_generation": question_metrics,
        "distractor_generation": distractor_metrics,
        "hint_generation": hint_metrics,
        "num_samples": len(sample_df),
        "samples": samples[:10],
    }

    save_json(config.MODEL_B_RESULTS_FILE, result)
    safe_save(config.MODEL_B_CHECKPOINT_DIR / "model_b_samples.joblib", samples)
    safe_save(config.MODEL_B_CHECKPOINT_DIR / "hint_eval.joblib", hint_metrics)
    safe_save(config.MODEL_B_CHECKPOINT_DIR / "distractor_eval.joblib", distractor_metrics)
    return result


def main() -> int:
    args = parse_args()
    config = Config(base_dir=args.base_dir)
    config.kmeans_sample_size = args.sample_size
    ensure_project_dirs(config)
    set_random_seed(config.random_seed)

    if args.retrain:
        maybe_remove_existing_models(config)

    try:
        prepared = prepare_processed_data(config)
    except FileNotFoundError:
        print("Dataset not found.")
        print("Place a source CSV under data/raw or follow the Kaggle download steps in README.")
        return 1
    except Exception as exc:
        print(f"Failed to prepare dataset: {exc}")
        return 1

    save_split_checkpoints(config, prepared)
    eda_summary = run_eda(
        prepared["raw_df"],
        prepared["clean_df"],
        prepared["train_df"],
        prepared["val_df"],
        prepared["test_df"],
        prepared["train_opt"],
        prepared["val_opt"],
        prepared["test_opt"],
        config,
    )
    print(f"EDA complete. Clean rows: {eda_summary['clean_row_count']}")

    label_mean = prepared["train_opt"]["label"].mean()
    print(f"Option-level positive rate: {label_mean:.4f}")

    model_a_results, trained_models = train_and_evaluate_model_a(
        config,
        prepared["train_opt"],
        prepared["val_opt"],
        prepared["test_opt"],
        quick=args.quick,
    )
    save_sparse_feature_checkpoints(config, trained_models, prepared)
    print("Model A training and evaluation complete.")

    unsupervised_results = train_unsupervised_component(config, prepared["train_opt"], quick=args.quick)
    print("Unsupervised component complete.")

    model_b_results = evaluate_model_b(config, prepared["test_df"], quick=args.quick)
    print("Model B evaluation complete.")

    best_model_name, best_model_metrics = max(
        model_a_results["models"].items(),
        key=lambda item: item[1]["question_level"]["question_level_accuracy"],
    )
    final_metrics = {
        "eda_summary_path": str(config.EDA_SUMMARY_FILE),
        "best_model_name": best_model_name,
        "best_question_accuracy": best_model_metrics["question_level"]["question_level_accuracy"],
        "best_option_macro_f1": best_model_metrics["option_level"]["macro_f1"],
        "best_positive_f1": best_model_metrics["option_level"]["positive_class_f1"],
        "ensemble_question_accuracy": model_a_results["models"]["ensemble"]["question_level"]["question_level_accuracy"],
        "model_a_best_question_accuracy": best_model_metrics["question_level"]["question_level_accuracy"],
        "unsupervised_cluster_purity_mean": unsupervised_results["cluster_purity_mean"],
        "model_b_question_bleu": model_b_results["question_generation"]["bleu"],
        "model_b_distractor_rougeL": model_b_results["distractor_generation"]["rougeL_f1"],
        "model_b_hint_meteor": model_b_results["hint_generation"]["meteor"],
    }
    save_final_metrics(config.FINAL_METRICS_FILE, final_metrics)
    print("Final metrics saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
