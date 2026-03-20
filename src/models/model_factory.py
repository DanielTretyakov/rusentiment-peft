"""
Фабрика моделей — единая точка входа для всех методов дообучения.
Поддерживаемые методы: full_finetune | lora | adapter | prefix_tuning
"""
from transformers import AutoModelForSequenceClassification
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    TaskType,
)


def build_model(config: dict):
    """
    Создаёт и возвращает модель на основе метода, указанного в конфиге.

    Аргументы:
        config: распарсенный словарь из YAML-конфига

    Возвращает:
        модель, готовую к обучению
    """
    method = config.get("method", "full_finetune")
    model_name = config["model"]["name"]
    num_labels = config["model"]["num_labels"]

    # Загружаем базовую предобученную модель
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    if method == "full_finetune":
        # Все параметры обучаемы — ничего не замораживаем
        return model

    elif method == "lora":
        peft_cfg = config.get("peft", {})
        lora_config = LoraConfig(
            r=peft_cfg.get("r", 16),                           # ранг матриц разложения
            lora_alpha=peft_cfg.get("lora_alpha", 32),         # коэффициент масштабирования
            lora_dropout=peft_cfg.get("lora_dropout", 0.1),
            target_modules=peft_cfg.get("target_modules", ["query", "value"]),
            bias=peft_cfg.get("bias", "none"),
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    elif method == "prefix_tuning":
        peft_cfg = config.get("peft", {})
        prefix_config = PrefixTuningConfig(
            num_virtual_tokens=peft_cfg.get("num_virtual_tokens", 20),  # кол-во виртуальных токенов
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, prefix_config)
        model.print_trainable_parameters()
        return model

    elif method == "adapter":
        # Adapter Tuning через LoRA с малым рангом в качестве приближения
        # Для полноценных адаптеров используйте библиотеку adapter-transformers
        peft_cfg = config.get("peft", {})
        adapter_config = LoraConfig(
            r=peft_cfg.get("adapter_size", 64) // 4,
            lora_alpha=peft_cfg.get("adapter_size", 64) // 2,
            lora_dropout=0.1,
            target_modules=["query", "key", "value", "dense"],
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, adapter_config)
        model.print_trainable_parameters()
        return model

    else:
        raise ValueError(
            f"Неизвестный метод: '{method}'. "
            f"Допустимые значения: full_finetune, lora, adapter, prefix_tuning"
        )
