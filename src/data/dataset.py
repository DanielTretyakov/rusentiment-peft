"""
Класс датасета RuSentiment.
Будет реализован на шаге 2 проекта.
"""
from torch.utils.data import Dataset


class RuSentimentDataset(Dataset):
    """Заглушка — будет реализована на шаге 2."""

    def __init__(self, path: str, tokenizer, max_length: int = 128):
        # TODO: реализовать на шаге 2
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError("Будет реализовано на шаге 2")
