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

    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ):
        """
        Аргументы:
            path:       путь к CSV-файлу (колонки: text, label_id)
            tokenizer:  токенизатор HuggingFace
            max_length: максимальная длина токенизированной последовательности
        """
        self.tokenizer  = tokenizer
        self.max_length = max_length

        df = pd.read_csv(path)
        self.texts  = df["text"].tolist()
        self.labels = df["label_id"].tolist()

        print(f"Загружено {len(self.texts)} примеров из {path}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Возвращает один токенизированный пример в виде словаря тензоров.

        Ключи: input_ids, attention_mask, labels
        """
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",      # дополняем до max_length
            truncation=True,           # обрезаем если длиннее max_length
            return_tensors="pt",       # возвращаем PyTorch тензоры
        )

        return {
            # squeeze убирает лишнее измерение батча (1, seq_len) → (seq_len,)
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_datasets(
    tokenizer: PreTrainedTokenizerBase,
    train_path: str,
    val_path: str,
    test_path: str,
    max_length: int = 128,
) -> tuple[RuSentimentDataset, RuSentimentDataset, RuSentimentDataset]:
    """
    Удобная функция для создания всех трёх сплитов сразу.

    Возвращает: (train_dataset, val_dataset, test_dataset)
    """
    print("Инициализация датасетов...")
    train_dataset = RuSentimentDataset(train_path, tokenizer, max_length)
    val_dataset   = RuSentimentDataset(val_path,   tokenizer, max_length)
    test_dataset  = RuSentimentDataset(test_path,  tokenizer, max_length)

    print(
        f"\nРазмеры датасетов:\n"
        f"  train: {len(train_dataset)}\n"
        f"  val:   {len(val_dataset)}\n"
        f"  test:  {len(test_dataset)}\n"
    )

    return train_dataset, val_dataset, test_dataset
