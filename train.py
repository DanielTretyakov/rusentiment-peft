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
from src.data.dataset import build_datasets


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

    # --- Загрузка датасетов ---
    data_cfg = config["data"]
    train_dataset, val_dataset, _ = build_datasets(
        tokenizer=tokenizer,
        train_path=data_cfg["train_path"],
        val_path=data_cfg["val_path"],
        test_path=data_cfg["test_path"],
        max_length=config["model"].get("max_length", 128),
    )

    print(f"\nСборка модели, метод: {config.get('method', 'full_finetune')}")
    model = build_model(config)
    model.to(device)

    # --- Запуск обучения ---
    print("\nЗапуск обучения...\n")
    trainer = run_training(config, model, train_dataset, val_dataset)

    # Сохраняем лучшую модель
    save_path = config["training"]["output_dir"]
    trainer.save_model(save_path)
    print(f"\nОбучение завершено. Модель сохранена в: {save_path}")


if __name__ == "__main__":
    main()
