import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# Названия классов тональности
LABEL_NAMES = ["negative", "neutral", "positive"]


def compute_metrics(eval_pred) -> dict:
    """
    Вызывается HuggingFace Trainer после каждого шага валидации.
    Возвращает accuracy, взвешенный и макро F1.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "f1_macro": f1_score(labels, predictions, average="macro"),
    }


def full_report(labels, predictions) -> str:
    """Полный отчёт sklearn по классам — используется в evaluate.py."""
    return classification_report(
        labels,
        predictions,
        target_names=LABEL_NAMES,
        digits=4,
    )


def get_confusion_matrix(labels, predictions) -> np.ndarray:
    """Возвращает матрицу ошибок."""
    return confusion_matrix(labels, predictions)
