# Report Notes

## Main idea

This project solves reading comprehension in two parts:

- Model A verifies answer options using classical ML on option-level text.
- Model B produces question, distractor, and hint outputs using rule-based NLP and simple scoring.

## Why article text is repeated in `combined_text`

The original notebook gave the article extra weight. The cleaned project keeps that idea in a simple, transparent form:

`combined_text = article + " " + article + " " + question + " " + option_text`

This is still classical feature engineering and worked well enough for a student-friendly baseline.

## Why question-level accuracy matters

Option-level accuracy can look misleading because each question creates 4 rows and only 1 is positive. A model can score around 75% by predicting many negatives. That is why the final project reports:

- positive-class F1
- confusion matrix
- question-level accuracy after comparing A, B, C, and D for each question

## Unsupervised component

MiniBatchKMeans is used on a TF-IDF sample of option-level rows. The report should discuss:

- cluster distribution
- majority label per cluster
- average cluster purity
- silhouette score when available

## Generation evaluation

Generated text tasks use:

- BLEU
- ROUGE
- METEOR

This follows the updated teacher guidance better than using only accuracy or precision.
