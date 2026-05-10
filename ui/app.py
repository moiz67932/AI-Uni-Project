from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import Config
from src.inference import generate_hints, predict_correct_option, run_full_pipeline, verify_answer
from src.utils import load_json, normalize_whitespace


st.set_page_config(page_title="AI Reading Comprehension System", layout="wide")


@st.cache_resource
def get_config() -> Config:
    try:
        base_dir = st.secrets.get("AI_RACE_PROJECT_BASE_DIR", str(ROOT_DIR))
    except Exception:
        base_dir = str(ROOT_DIR)
    return Config(base_dir=base_dir)


@st.cache_data
def load_dataset(config: Config) -> pd.DataFrame:
    if config.TEST_CSV.exists():
        return pd.read_csv(config.TEST_CSV)
    if config.DATA_FILE.exists():
        return pd.read_csv(config.DATA_FILE)
    return pd.DataFrame()


def choose_random_sample(df: pd.DataFrame) -> dict | None:
    if df.empty:
        return None
    sample = df.sample(1, random_state=None).iloc[0]
    return sample.to_dict()


config = get_config()
dataset_df = load_dataset(config)

if "current_row" not in st.session_state:
    st.session_state.current_row = None
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "article_text" not in st.session_state:
    st.session_state.article_text = ""
if "used_all_hints" not in st.session_state:
    st.session_state.used_all_hints = False

st.sidebar.title("Controls")
model_name = st.sidebar.selectbox(
    "Model selection",
    [
        "random_forest",
        "ensemble",
        "tfidf_numeric_lr",
        "feature_logistic_regression",
        "count_lr",
        "tfidf_lr",
        "tfidf_svm",
    ],
)
data_source = st.sidebar.radio("Data source", ["Random RACE sample", "Paste custom article"])

if st.sidebar.button("Load random sample"):
    row = choose_random_sample(dataset_df)
    st.session_state.current_row = row
    if row:
        st.session_state.article_text = normalize_whitespace(row.get("article", ""))
        st.session_state.current_result = None

if st.sidebar.button("Run pipeline"):
    row = st.session_state.current_row if data_source == "Random RACE sample" else None
    article_text = st.session_state.article_text
    with st.spinner("Running the classical ML pipeline..."):
        st.session_state.current_result = run_full_pipeline(article_text, row=row, config=config, model_name=model_name)

if st.sidebar.button("Reset"):
    st.session_state.current_row = None
    st.session_state.current_result = None
    st.session_state.article_text = ""
    st.session_state.used_all_hints = False

tab1, tab2, tab3, tab4 = st.tabs(
    ["Article Input", "Question and Answer Quiz View", "Hint Panel", "Developer or Analytics Dashboard"]
)

with tab1:
    st.subheader("Article Input")
    if data_source == "Random RACE sample" and st.session_state.current_row:
        st.info("Using a random sample from the dataset.")
    st.session_state.article_text = st.text_area(
        "Article",
        value=st.session_state.article_text,
        height=250,
        placeholder="Paste an article here or load a random RACE sample.",
    )
    if data_source == "Paste custom article" and len(normalize_whitespace(st.session_state.article_text)) < 50:
        st.warning("Custom article looks short. A longer passage gives better questions, hints, and distractors.")
    if not config.DATA_FILE.exists() and dataset_df.empty:
        st.warning("Dataset not found. Please place race_single.csv in data/raw or run the Kaggle download step.")

with tab2:
    st.subheader("Question and Answer Quiz View")
    result = st.session_state.current_result
    if not result:
        st.info("Run the pipeline to see a question and answer options.")
    else:
        question = result["question"]
        options = result["options"]
        st.write(question)
        selected = st.radio("Choose your answer", ["A", "B", "C", "D"], format_func=lambda key: f"{key}: {options.get(key, '')}")
        if st.button("Check answer"):
            check = verify_answer(
                result["article"],
                question,
                options,
                selected,
                config=config,
                model_name=model_name,
            )
            if check["is_correct"]:
                st.success(f"Correct according to Model A. Confidence: {check['confidence']:.3f}")
            else:
                st.error(
                    f"Model A predicts {check['predicted_option']} instead. Confidence: {check['confidence']:.3f}"
                )
            st.write("Option probabilities:", check["option_probabilities"])
        if result["prediction"].get("error"):
            st.error(result["prediction"]["error"])
        elif result["prediction"].get("predicted_option"):
            st.caption(
                f"Predicted best option: {result['prediction']['predicted_option']} "
                f"(confidence {result['prediction']['confidence']:.3f})"
            )

with tab3:
    st.subheader("Hint Panel")
    result = st.session_state.current_result
    if not result:
        st.info("Run the pipeline to generate hints.")
    else:
        hints = result["hints"]
        for index, hint in enumerate(hints, start=1):
            with st.expander(f"Hint {index}"):
                st.write(hint)
        if st.button("I used all hints"):
            st.session_state.used_all_hints = True
        if st.session_state.used_all_hints:
            correct_option = result.get("correct_option", "")
            if correct_option:
                st.success(
                    f"Reveal answer: {correct_option} - {result['options'].get(correct_option, '')}"
                )
            else:
                st.info("No known answer was available for this custom article demo.")

with tab4:
    st.subheader("Developer or Analytics Dashboard")
    model_a_metrics = load_json(config.MODEL_A_RESULTS_FILE, default={})
    model_b_metrics = load_json(config.MODEL_B_RESULTS_FILE, default={})
    final_metrics = load_json(config.FINAL_METRICS_FILE, default={})

    if model_a_metrics:
        st.write("Model A metrics")
        st.json(model_a_metrics)
    else:
        st.warning("Model files not found. Please run training first.")

    if model_b_metrics:
        st.write("Model B metrics")
        st.json(model_b_metrics)

    if config.CONFUSION_MATRIX_FIG.exists():
        st.image(str(config.CONFUSION_MATRIX_FIG), caption="Confusion matrix")

    if st.session_state.current_result and st.session_state.current_result["prediction"]:
        st.metric("Last inference latency", st.session_state.current_result["prediction"].get("latency_seconds", 0.0))

    if config.INFERENCE_LOG_FILE.exists():
        logs_df = pd.read_csv(config.INFERENCE_LOG_FILE)
        st.write("Recent inference logs")
        st.dataframe(logs_df.tail(20), use_container_width=True)
        st.download_button("Export logs as CSV", logs_df.to_csv(index=False), file_name="inference_logs.csv")
    else:
        st.info("No inference logs yet.")

    if final_metrics:
        st.write("Final metrics summary")
        st.json(final_metrics)
