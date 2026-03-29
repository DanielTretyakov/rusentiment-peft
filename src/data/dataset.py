"""
PyTorch Dataset для датасета RuSentiment.

Принимает CSV-файл с колонками 'text' и 'label_id',
токенизирует тексты и возвращает тензоры для HuggingFace Trainer.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class RuSentimentDataset(Dataset):
    """
    Датасет для задачи классификации тональности.

    Пример использования:
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        dataset = RuSentimentDataset("data/processed/train.csv", tokenizer)
        sample = dataset[0]
        # sample содержит: input_ids, attention_mask, labels
    """

    def __init__(self, path, tokenizer, max_length=128, max_samples=None):
        """
        Аргументы:
            path:        путь к CSV-файлу (колонки: text, label_id)
            tokenizer:   токенизатор HuggingFace
            max_length:  максимальная длина токенизированной последовательности
            max_samples: если задано, берём только первые N примеров (для быстрых экспериментов)
        """
        self.tokenizer  = tokenizer
        self.max_length = max_length

        df = pd.read_csv(path)

        # Стратифицированная выборка — сохраняем баланс классов
        if max_samples is not None and max_samples < len(df):
            df = (
                df.groupby("label_id", group_keys=False)
                .apply(lambda x: x.sample(max_samples // 3, random_state=42))
                .reset_index(drop=True)
            )
            print("Используем " + str(len(df)) + " примеров из " + path)
        else:
            print("Загружено " + str(len(df)) + " примеров из " + path)

        self.texts  = df["text"].tolist()
        self.labels = df["label_id"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Возвращает один токенизированный пример в виде словаря тензоров.

        Ключи: input_ids, attention_mask, labels
        """
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            # squeeze убирает лишнее измерение батча (1, seq_len) -> (seq_len,)
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_datasets(tokenizer, train_path, val_path, test_path, max_length=128, max_samples=None):
    """
    Удобная функция для создания всех трёх сплитов сразу.

    Аргументы:
        max_samples: максимум примеров на train (val и test берутся полностью)

    Возвращает: (train_dataset, val_dataset, test_dataset)
    """
    print("Инициализация датасетов...")
    train_dataset = RuSentimentDataset(train_path, tokenizer, max_length, max_samples=max_samples)
    val_dataset   = RuSentimentDataset(val_path,   tokenizer, max_length)
    test_dataset  = RuSentimentDataset(test_path,  tokenizer, max_length)

    print(
        "\nРазмеры датасетов:\n"
        "  train: " + str(len(train_dataset)) + "\n"
        "  val:   " + str(len(val_dataset))   + "\n"
        "  test:  " + str(len(test_dataset))  + "\n"
    )

    return train_dataset, val_dataset, test_dataset
