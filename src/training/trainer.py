"""
Обёртка над HuggingFace Trainer.
Отвечает за настройку аргументов обучения и запуск цикла обучения.
Логирование ведётся локально (без W&B).
"""
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from src.training.metrics import compute_metrics
from src.utils.utils import ensure_dir, count_parameters


def build_training_args(config):
    """Формирует TrainingArguments из словаря конфига."""
    t = config["training"]
    ensure_dir(t["output_dir"])

    return TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t.get("num_epochs", 5),
        per_device_train_batch_size=t.get("batch_size", 16),
        per_device_eval_batch_size=t.get("batch_size", 16),
        learning_rate=t.get("learning_rate", 2e-5),
        weight_decay=t.get("weight_decay", 0.01),
        warmup_ratio=t.get("warmup_ratio", 0.1),
        eval_strategy="steps",
        eval_steps=t.get("eval_steps", 200),
        save_strategy="steps",
        save_steps=t.get("save_steps", 200),
        logging_steps=t.get("log_steps", 50),
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        fp16=t.get("fp16", True),
        seed=t.get("seed", 42),
        report_to="none",
    )


def run_training(config, model, train_dataset, val_dataset):
    """
    Создаёт Trainer и запускает обучение.

    Аргументы:
        config:        полный словарь конфига
        model:         модель из model_factory.build_model()
        train_dataset: токенизированный датасет для обучения
        val_dataset:   токенизированный датасет для валидации

    Возвращает:
        обученный объект Trainer
    """
    # Вывод статистики по параметрам
    param_info = count_parameters(model)
    print(
        "\n" + "="*50 + "\n"
        "Метод:                " + config.get("method", "unknown") + "\n"
        "Всего параметров:     " + str(param_info["total"]) + "\n"
        "Обучаемых параметров: " + str(param_info["trainable"]) +
        " (" + str(param_info["trainable_pct"]) + "%)\n" +
        "="*50 + "\n"
    )

    # Создание Trainer
    training_args = build_training_args(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    return trainer
