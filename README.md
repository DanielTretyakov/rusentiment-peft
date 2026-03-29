# 🔬 RuSentiment PEFT

> Comparative study of **Parameter-Efficient Fine-Tuning** methods for Russian sentiment analysis.

[![CI](https://github.com/YOUR_USERNAME/rusentiment-peft/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/rusentiment-peft/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange)](https://pytorch.org/)

---

## 🎯 Overview

This project benchmarks four fine-tuning strategies on the [RuSentiment](https://www.kaggle.com/datasets/mar1mba/russian-sentiment-dataset) dataset using **DeepPavlov/rubert-base-cased** as the backbone model.

**Task:** 3-class Russian sentiment classification — `negative` / `neutral` / `positive`

**Dataset:** 290,456 texts, balanced (~33% per class), split 80/10/10

---

## 📊 Results

| Method | Accuracy | F1 Weighted | F1 Macro | Trainable Params | Trainable % |
|--------|----------|-------------|----------|-----------------|-------------|
| **Adapter Tuning** | **70.73%** | **70.78%** | **70.77%** | ~1.2M | ~0.67% |
| Full Fine-Tuning | 69.70% | 70.13% | 70.13% | 178.4M | 100% |
| LoRA | 68.91% | 68.38% | 68.37% | 592K | 0.33% |
| Prefix Tuning | 61.22% | 59.26% | 59.24% | 371K | 0.21% |

### Per-class F1 scores

| Method | negative | neutral | positive |
|--------|----------|---------|----------|
| Adapter Tuning | 0.6078 | 0.8078 | 0.7075 |
| Full Fine-Tuning | 0.6208 | 0.7902 | 0.6929 |
| LoRA | 0.5437 | 0.7946 | 0.7126 |
| Prefix Tuning | 0.4089 | 0.7155 | 0.6528 |

### Key findings

- **Adapter Tuning outperforms Full Fine-Tuning** while training only 0.67% of parameters — the main finding of this study
- **LoRA achieves competitive results** with just 0.33% trainable parameters, losing only ~1.7% F1 vs full fine-tuning
- **Prefix Tuning underperforms** significantly, especially on the `negative` class (F1=0.41)
- **Neutral class** is consistently the easiest to classify across all methods (F1 ~0.79–0.81)
- **Negative class** is the hardest — typical for Russian-language sentiment datasets

---

## 🧠 Methods Explained

### LoRA (Low-Rank Adaptation)
Freezes all original weights. Injects trainable rank-decomposition matrices **A** and **B** into attention layers. The weight update is `ΔW = BA` where rank `r << d`. Extremely parameter-efficient with minimal quality loss.

### Adapter Tuning
Inserts small bottleneck feed-forward modules after each transformer sub-layer. Only adapter parameters (~64-dim bottleneck) are trained. Showed the best results in this study.

### Prefix Tuning
Prepends a set of learnable virtual tokens to the key and value matrices of every attention layer. The original model is completely frozen. Significantly underperformed other methods on this task.

### Full Fine-Tuning (Baseline)
All 178M parameters are updated during training. Serves as the upper-bound baseline. Notably outperformed by Adapter Tuning despite training 150x fewer parameters.

---

## 🗂️ Project Structure

```
rusentiment-peft/
├── configs/                  # YAML experiment configs
│   ├── base.yaml
│   ├── lora.yaml
│   ├── adapter.yaml
│   ├── prefix_tuning.yaml
│   └── full_finetune.yaml
├── src/
│   ├── data/
│   │   ├── dataset.py        # PyTorch Dataset + tokenization
│   │   └── preprocessing.py  # Data cleaning + train/val/test split
│   ├── models/
│   │   └── model_factory.py  # Single entry point for all PEFT methods
│   ├── training/
│   │   ├── trainer.py        # HuggingFace Trainer wrapper
│   │   └── metrics.py        # Accuracy, F1 weighted, F1 macro
│   └── utils/
│       └── utils.py          # Seed, device, parameter counting
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   └── 02_results.ipynb      # Results visualization & comparison
├── experiments/              # Saved checkpoints & metrics
├── tests/                    # Unit tests (pytest)
├── train.py                  # Training entry point
└── evaluate.py               # Evaluation entry point
```

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/rusentiment-peft.git
cd rusentiment-peft

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle and place CSV in data/raw/rusentiment.csv

# 4. Preprocess data
python -c "from src.data.preprocessing import prepare_data; prepare_data('data/raw/rusentiment.csv')"

# 5. Train
python train.py --config configs/lora.yaml
python train.py --config configs/adapter.yaml
python train.py --config configs/prefix_tuning.yaml
python train.py --config configs/full_finetune.yaml

# 6. Evaluate on test set
python evaluate.py --config configs/lora.yaml --checkpoint experiments/lora/checkpoint-XXXX
```

Or use Makefile shortcuts:

```bash
make train-lora
make train-adapter
make train-prefix
make train-full
```

---

## 🧪 Tests

```bash
make test
# or: pytest tests/ -v
```

---

## 📚 References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) — Li & Liang, 2021
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) — Houlsby et al., 2019
- [HuggingFace PEFT library](https://github.com/huggingface/peft)
- [DeepPavlov/rubert-base-cased](https://huggingface.co/DeepPavlov/rubert-base-cased)

---

## 👤 Author

**Your Name** — [GitHub](https://github.com/YOUR_USERNAME)
