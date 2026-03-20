"""
Тесты для модулей предобработки и датасета.
"""
import pytest
import pandas as pd
import torch
from unittest.mock import MagicMock

from src.data.preprocessing import clean_text, load_and_clean, split_dataset, LABEL2ID
from src.data.dataset import RuSentimentDataset


# ──────────────────────────────────────────
# Тесты clean_text
# ──────────────────────────────────────────

def test_clean_text_removes_url():
    text = "Смотри тут https://vk.com/wall123 классное видео"
    result = clean_text(text)
    assert "https" not in result
    assert "классное видео" in result


def test_clean_text_removes_mention():
    text = "@username привет как дела"
    result = clean_text(text)
    assert "@username" not in result
    assert "привет" in result


def test_clean_text_collapses_punctuation():
    text = "Это невероятно!!!!!!"
    result = clean_text(text)
    assert "!!!!!!" not in result
    assert "!" in result


def test_clean_text_handles_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""


# ──────────────────────────────────────────
# Тесты load_and_clean
# ──────────────────────────────────────────

def test_load_and_clean_filters_labels(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text(
        "text,label\n"
        "хорошо,positive\n"
        "плохо,negative\n"
        "нейтрально,neutral\n"
        "пропустить,skip\n"
        "речь,speech\n"
    )
    df = load_and_clean(str(csv))
    # speech и skip должны быть отфильтрованы
    assert set(df["label"].unique()) == {"positive", "negative", "neutral"}
    assert len(df) == 3


def test_load_and_clean_adds_label_id(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text("text,label\nотлично,positive\nужасно,negative\n")
    df = load_and_clean(str(csv))
    assert "label_id" in df.columns
    assert df[df["label"] == "positive"]["label_id"].iloc[0] == LABEL2ID["positive"]


# ──────────────────────────────────────────
# Тесты split_dataset
# ──────────────────────────────────────────

def test_split_dataset_sizes():
    # Создаём сбалансированный мини-датасет
    df = pd.DataFrame({
        "text":     [f"текст {i}" for i in range(300)],
        "label":    ["positive"] * 100 + ["negative"] * 100 + ["neutral"] * 100,
        "label_id": [2] * 100 + [0] * 100 + [1] * 100,
    })
    train_df, val_df, test_df = split_dataset(df, val_size=0.1, test_size=0.1)
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == 300
    assert len(test_df) == pytest.approx(30, abs=2)
    assert len(val_df)  == pytest.approx(30, abs=2)


# ──────────────────────────────────────────
# Тесты RuSentimentDataset
# ──────────────────────────────────────────

def test_dataset_returns_correct_keys(tmp_path):
    """Проверяем, что __getitem__ возвращает нужные тензоры."""
    csv = tmp_path / "train.csv"
    csv.write_text("text,label_id\nхорошая погода,2\nплохая погода,0\n")

    # Мокаем токенизатор — не нужно загружать реальную модель
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids":      torch.zeros(1, 128, dtype=torch.long),
        "attention_mask": torch.ones(1, 128, dtype=torch.long),
    }

    dataset = RuSentimentDataset(str(csv), mock_tokenizer, max_length=128)
    assert len(dataset) == 2

    sample = dataset[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert sample["labels"].item() == 2
