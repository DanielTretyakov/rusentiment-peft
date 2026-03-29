"""
Точка входа для оценки обученных моделей на тестовой выборке.

Использование:
    python evaluate.py --config configs/lora.yaml --checkpoint experiments/lora/checkpoint-1800
    python evaluate.py --config configs/full_finetune.yaml --checkpoint experiments/full_finetune/checkpoint-XXXX
"""
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from src.utils.utils import load_config, set_seed, get_device
from src.data.dataset import RuSentimentDataset
from src.training.metrics import full_report, get_confusion_matrix

# Названия классов для вывода
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def parse_args():
    parser = argparse.ArgumentParser(description="Оценка обученной модели на тестовой выборке")
    parser.add_argument("--config",     type=str, required=True,
                        help="Путь к YAML-конфигу (например: configs/lora.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Путь к чекпоинту (например: experiments/lora/checkpoint-1800)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Размер батча при инференсе (по умолчанию: 32)")
    return parser.parse_args()


def load_model(config, checkpoint_path, device):
    """
    Загружает модель из чекпоинта.
    Для LoRA и Adapter сливает с базовой моделью.
    Для Prefix Tuning оставляет адаптер в модели.
    """
    method     = config.get("method", "full_finetune")
    model_name = config["model"]["name"]
    num_labels = config["model"]["num_labels"]

    print("Загрузка базовой модели: " + model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    if method == "full_finetune":
        print("Загрузка весов full_finetune из: " + checkpoint_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            num_labels=num_labels,
        )
    else:
        print("Загрузка PEFT-адаптера из: " + checkpoint_path)
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
        # LoRA и Adapter поддерживают merge_and_unload
        # Prefix Tuning не поддерживает
        if method in ["lora", "adapter"]:
            print("Слияние весов адаптера с базовой моделью...")
            model = model.merge_and_unload()
        else:
            print(f"Адаптер {method} будет использован вместе с базовой моделью")

    model.to(device)
    model.eval()
    return model


def run_inference(model, dataloader, device):
    """
    Запускает инференс на всём датасете.
    Возвращает массивы предсказаний и истинных меток.
    """
    all_preds  = []
    all_labels = []

    print("Запуск инференса...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

            # Прогресс каждые 50 батчей
            if (i + 1) % 50 == 0:
                print("  Обработано батчей: " + str(i + 1) + " / " + str(len(dataloader)))

    return np.array(all_labels), np.array(all_preds)


def save_results(results, output_path):
    """Сохраняет результаты оценки в JSON-файл."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Результаты сохранены в: " + output_path)


def main():
    args   = parse_args()
    config = load_config(args.config)
    method = config.get("method", "full_finetune")

    set_seed(config["training"].get("seed", 42))
    device = get_device()

    # Загрузка токенизатора
    print("Загрузка токенизатора: " + config["model"]["name"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    # Загрузка тестового датасета (полностью, без max_samples)
    test_path = config["data"]["test_path"]
    print("Загрузка тестового датасета: " + test_path)
    test_dataset = RuSentimentDataset(
        path=test_path,
        tokenizer=tokenizer,
        max_length=config["model"].get("max_length", 128),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Загрузка модели
    model = load_model(config, args.checkpoint, device)

    # Инференс
    labels, preds = run_inference(model, test_dataloader, device)

    # Метрики
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ — метод: " + method)
    print("="*60)
    print(full_report(labels, preds))

    # Матрица ошибок
    cm = get_confusion_matrix(labels, preds)
    print("Матрица ошибок (строки=истина, столбцы=предсказание):")
    print("              " + "  ".join([ID2LABEL[i].ljust(10) for i in range(3)]))
    for i, row in enumerate(cm):
        print(ID2LABEL[i].ljust(14) + "  ".join([str(v).ljust(10) for v in row]))

    # Сохраняем результаты в JSON
    from sklearn.metrics import accuracy_score, f1_score
    results = {
        "method":       method,
        "checkpoint":   args.checkpoint,
        "test_samples": len(labels),
        "accuracy":     round(accuracy_score(labels, preds), 4),
        "f1_weighted":  round(f1_score(labels, preds, average="weighted"), 4),
        "f1_macro":     round(f1_score(labels, preds, average="macro"), 4),
        "f1_per_class": {
            ID2LABEL[i]: round(f1_score(labels, preds, average=None)[i], 4)
            for i in range(3)
        },
        "confusion_matrix": cm.tolist(),
    }

    output_path = os.path.join(
        config["training"]["output_dir"],
        "test_results.json"
    )
    save_results(results, output_path)

    print("\nИтого:")
    print("  Accuracy:    " + str(results["accuracy"]))
    print("  F1 weighted: " + str(results["f1_weighted"]))
    print("  F1 macro:    " + str(results["f1_macro"]))


if __name__ == "__main__":
    main()
