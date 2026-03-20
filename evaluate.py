"""
Точка входа для оценки обученной модели на тестовой выборке.

Использование:
    python evaluate.py --config configs/lora.yaml --checkpoint experiments/lora/
"""
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from src.utils.utils import load_config, set_seed, get_device
from src.training.metrics import full_report, get_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Оценка обученной PEFT-модели")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Путь к директории с сохранённым чекпоинтом")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["training"].get("seed", 42))
    device = get_device()

    method = config.get("method", "full_finetune")
    model_name = config["model"]["name"]

    print(f"Загрузка токенизатора: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Загрузка модели из чекпоинта: {args.checkpoint}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config["model"]["num_labels"],
    )

    if method != "full_finetune":
        # Загружаем PEFT-адаптер и сливаем веса для инференса
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
        model = model.merge_and_unload()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)

    model.to(device)
    model.eval()

    # --- Заглушка: тестовый датасет будет подключён на шаге 2 ---
    print("\n[!] Загрузка тестового датасета ещё не реализована — будет добавлена на шаге 2.\n")

    # Так будет выглядеть полный цикл оценки:
    # all_preds, all_labels = [], []
    # for batch in test_dataloader:
    #     with torch.no_grad():
    #         outputs = model(**batch)
    #     preds = np.argmax(outputs.logits.cpu().numpy(), axis=-1)
    #     all_preds.extend(preds)
    #     all_labels.extend(batch["labels"].cpu().numpy())
    #
    # print(full_report(all_labels, all_preds))
    # print(get_confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
