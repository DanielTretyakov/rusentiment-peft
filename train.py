"""
Точка входа для обучения модели.

Использование:
    python train.py --config configs/lora.yaml
    python train.py --config configs/full_finetune.yaml
"""
import argparse
from transformers import AutoTokenizer

from src.utils.utils import load_config, set_seed, get_device
from src.models.model_factory import build_model
from src.training.trainer import run_training

# ПРИМЕЧАНИЕ: загрузка датасета будет подключена на шаге 2
# from src.data.dataset import RuSentimentDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение PEFT-модели на датасете RuSentiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Путь к YAML-конфигу (например: configs/lora.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["training"].get("seed", 42))
    device = get_device()

    print(f"\nЗагрузка токенизатора: {config['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    print(f"Сборка модели, метод: {config.get('method', 'full_finetune')}")
    model = build_model(config)
    model.to(device)

    # --- Заглушка: датасеты будут загружены здесь на шаге 2 ---
    # train_dataset = RuSentimentDataset(config["data"]["train_path"], tokenizer, ...)
    # val_dataset   = RuSentimentDataset(config["data"]["val_path"], tokenizer, ...)
    print("\n[!] Загрузка датасета ещё не реализована — будет добавлена на шаге 2.\n")

    # trainer = run_training(config, model, train_dataset, val_dataset)
    # trainer.save_model(config["training"]["output_dir"])
    # print("Обучение завершено. Модель сохранена.")


if __name__ == "__main__":
    main()
