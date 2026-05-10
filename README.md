# Intelligent Reading Comprehension and Quiz Generation System using Classical Machine Learning

This FAST BS(CS) Spring 2026 semester project builds a reading comprehension and quiz generation system with classical machine learning and rule-based NLP. The main answer verification module uses sklearn pipelines instead of neural networks, and the generation side uses template-based and heuristic methods so the project stays within the teacher's rules.

## Why classical ML

The teacher clarified that neural networks are not allowed as the main solution. This project therefore uses:

- `CountVectorizer + LogisticRegression`
- `TfidfVectorizer + LogisticRegression`
- `TfidfVectorizer + calibrated LinearSVC`
- `RandomForest` on small handcrafted features
- `MiniBatchKMeans` as the unsupervised component
- Rule-based distractor and hint generation

No BERT, T5, transformers, PyTorch training, or TensorFlow training are used as the main system.

## Project structure

```text
MyDrive/
  AI_RACE_Project/
    data/
      raw/
        race_single.csv
      processed/
        train_80.csv
        val_10.csv
        test_10.csv
        train_option_level.csv
        val_option_level.csv
        test_option_level.csv
    checkpoints/
      model_a/
      model_b/
    models/
      model_a/traditional/
      model_b/traditional/
    notebooks/
    src/
    ui/
    outputs/
    report/
    requirements.txt
    README.md
    run_all.py
```

## Colab setup

Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Recommended project base folder:

```python
BASE_DIR = "/content/drive/MyDrive/AI_RACE_Project"
```

## Kaggle setup

Upload `kaggle.json` in Colab, then run:

```bash
mkdir -p ~/.kaggle
cp /content/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
export BASE_DIR="/content/drive/MyDrive/AI_RACE_Project"
kaggle datasets download -d ankitdhiman7/race-dataset -p "$BASE_DIR/data/raw" --unzip
```

Dataset rule used in this project:

- Only one source CSV is used for the final workflow.
- If `data/raw/race_single.csv` does not exist, the code looks for CSV files under `data/raw`.
- It prefers `train.csv`.
- Otherwise it picks the largest CSV that contains the required RACE columns.
- That file is copied to `data/raw/race_single.csv`.
- Final train, validation, and test sets are created by the project itself using `80/10/10`.

## Install

```bash
pip install -r requirements.txt
```

## How to train

Local:

```bash
python run_all.py
```

Quick smoke run:

```bash
python run_all.py --quick --sample-size 2000
```

Force retraining:

```bash
python run_all.py --retrain
```

If you want the script to write directly into Google Drive from Colab:

```bash
python run_all.py --base-dir /content/drive/MyDrive/AI_RACE_Project
```

## How to run the UI

```bash
streamlit run ui/app.py
```

## What gets saved

- Processed splits in `data/processed/`
- Checkpoints in `checkpoints/model_a/` and `checkpoints/model_b/`
- Trained models in `models/model_a/traditional/` and `models/model_b/traditional/`
- Metrics JSON files in `outputs/results/`
- Figures in `outputs/figures/`
- Inference logs in `outputs/logs/inference_logs.csv`

## Checkpoint safety and resume behavior

The project uses `joblib` for sklearn models and checkpoint objects. Atomic saving is used in `src/checkpointing.py`:

1. Save to a temporary file ending in `.tmp`
2. Replace the old file only after the dump succeeds

If loading fails because a checkpoint is corrupted:

- a friendly warning is printed
- the bad file is renamed with a `.corrupt` suffix
- the code returns a default value and retrains when needed

This is the fix for the old truncated `model_svm.pkl` style problem.

## Models implemented

Model A answer verification:

- Count LR
- TFIDF LR
- TFIDF SVM with calibration
- RandomForest on numeric features
- Feature Logistic Regression on numeric features
- TFIDF Numeric LR using sparse TF-IDF plus numeric features
- Simple soft-voting ensemble of the three text models
- Validation-weighted ensemble saved as `ensemble_model.joblib`
- MiniBatchKMeans unsupervised component

Model B generation:

- Template-based question generation
- Distractor generator using original wrong options when available and heuristic extraction otherwise
- Graduated hint generator using passage sentence similarity

## Metrics reported

Classification:

- accuracy
- macro F1
- weighted F1
- precision
- recall
- `positive_class_precision`
- `positive_class_recall`
- `positive_class_f1`
- confusion matrix
- TP, TN, FP, FN
- question-level accuracy

Generated text:

- BLEU
- ROUGE-1
- ROUGE-L
- METEOR

## Notebook usage

The project does not rely only on notebooks anymore. The new notebooks import from `src/` so the same code works in Colab and locally.

- `notebooks/00_eda.ipynb`
- `notebooks/01_model_a_training.ipynb`
- `notebooks/02_model_b_distractors_hints.ipynb`
- `notebooks/03_evaluation.ipynb`

`Project2_0.ipynb` is kept as the original backup base.

## Limitations

- Model B generation is template-based and heuristic, so it is useful for demos but not as fluent as neural text generation.
- Custom pasted articles may need manual option editing for the best quiz experience.
- KMeans is included for the rubric and interpretability, not because it outperforms supervised verification.

## Submission checklist

- `race_single.csv` prepared from one Kaggle source CSV
- `80/10/10` split saved
- classical Model A trained
- unsupervised component trained
- ensemble evaluated
- distractor and hint generation implemented
- BLEU, ROUGE, METEOR saved
- Streamlit UI runs
- README and requirements included
- corrupted checkpoints handled safely
