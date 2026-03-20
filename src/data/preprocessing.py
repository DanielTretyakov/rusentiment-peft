"""
Предобработка данных для датасета RuSentiment.

Формат файла:
  - Разделитель: ; (точка с запятой)
  - Кодировка: cp1251
  - Колонки: text, label, src
  - Метки: 0 (negative), 1 (neutral), 2 (positive)
"""

import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# Числовой индекс -> название класса
ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

# Допустимые метки
KEEP_LABELS = {0, 1, 2}


def clean_text(text):
    """
    Очистка одного текста.

    Шаги:
      1. Удаление URL-ссылок
      2. Удаление упоминаний (@user)
      3. Сжатие повторяющихся знаков препинания (!!!!! -> !)
      4. Удаление лишних пробелов и переносов строк
    """
    if not isinstance(text, str):
        return ""

    # Удаляем ссылки (http/https/www)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Удаляем упоминания вида @username
    text = re.sub(r"@\w+", "", text)

    # Сжимаем повторяющиеся знаки: !!!!! -> !
    text = re.sub(r"([!?.]){2,}", r"\1", text)

    # Убираем внутренние переносы строк (встречаются в текстах датасета)
    text = re.sub(r"[\r\n]+", " ", text)

    # Убираем лишние пробелы
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_and_clean(path):
    """
    Загружает CSV-файл датасета и очищает тексты.

    Возвращает DataFrame с колонками:
      - text:     очищенный текст
      - label_id: числовой индекс метки (0 / 1 / 2)
    """
    print("Чтение файла: " + path)
    df = pd.read_csv(
        path,
        sep=";",
        encoding="cp1251",
        encoding_errors="replace",
    )

    # Приводим названия колонок к нижнему регистру
    df.columns = [c.lower().strip() for c in df.columns]

    # Оставляем только нужные колонки
    df = df[["text", "label"]].copy()

    # Фильтруем строки с недопустимыми метками
    df = df[df["label"].isin(KEEP_LABELS)].copy()

    # Очищаем тексты
    print("Очистка текстов...")
    df["text"] = df["text"].apply(clean_text)

    # Удаляем строки с пустым текстом после очистки
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    # Переименовываем label -> label_id для единообразия с остальным кодом
    df = df.rename(columns={"label": "label_id"})

    return df[["text", "label_id"]]


def split_dataset(df, val_size=0.1, test_size=0.1, seed=42):
    """
    Стратифицированное разбиение датасета на train / val / test.

    Пропорции по умолчанию: 80% / 10% / 10%
    Стратификация гарантирует одинаковое распределение классов во всех сплитах.

    Возвращает: (train_df, val_df, test_df)
    """
    # Сначала отделяем тест
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label_id"],
        random_state=seed,
    )

    # Из оставшегося отделяем валидацию
    # val_size пересчитывается относительно train_val размера
    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["label_id"],
        random_state=seed,
    )

    # Сбрасываем индексы
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def prepare_data(raw_path, output_dir="data/processed", seed=42):
    """
    Полный пайплайн предобработки: загрузка -> очистка -> сплит -> сохранение.

    Использование:
        python -c "from src.data.preprocessing import prepare_data; prepare_data('data/raw/rusentiment.csv')"

    После запуска в data/processed/ появятся:
        train.csv, val.csv, test.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    df = load_and_clean(raw_path)

    print("Всего примеров после фильтрации: " + str(len(df)))
    print("Распределение классов:")
    counts = df["label_id"].value_counts().sort_index()
    for label_id, count in counts.items():
        print("  " + str(label_id) + " (" + ID2LABEL[label_id] + "): " + str(count))

    print("\nРазбиение на train / val / test (80/10/10)...")
    train_df, val_df, test_df = split_dataset(df, seed=seed)

    train_path = os.path.join(output_dir, "train.csv")
    val_path   = os.path.join(output_dir, "val.csv")
    test_path  = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    test_df.to_csv(test_path,   index=False)

    print("\nГотово! Файлы сохранены в '" + output_dir + "':")
    print("  train.csv - " + str(len(train_df)) + " примеров")
    print("  val.csv   - " + str(len(val_df))   + " примеров")
    print("  test.csv  - " + str(len(test_df))  + " примеров")
