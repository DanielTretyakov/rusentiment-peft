# 🔬 RuSentiment PEFT

> Comparative study of **Parameter-Efficient Fine-Tuning** methods for Russian sentiment analysis.

[![CI](https://github.com/YOUR_USERNAME/rusentiment-peft/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/rusentiment-peft/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-tracked-orange)](https://wandb.ai/)

---

## 🎯 Overview

This project benchmarks four fine-tuning strategies on the [RuSentiment](https://www.kaggle.com/datasets/mar1mba/russian-sentiment-dataset) dataset using **rubert-base-cased** as the backbone model:

| Method | Trainable Params | Description |
|--------|-----------------|-------------|
| Full Fine-Tuning | ~180M (100%) | All weights updated — strong baseline |
| **LoRA** | ~300K (0.17%) | Low-rank decomposition of attention matrices |
| Adapter Tuning | ~1.2M (0.67%) | Small bottleneck layers injected per block |
| Prefix Tuning | ~200K (0.11%) | Trainable prefix tokens prepended to each layer |

**Key research questions:**
- Can PEFT methods match full fine-tuning accuracy while training <1% of parameters?
- Which method offers the best accuracy / parameter trade-off?
- How do methods behave under limited compute?

---

## 📊 Results

> *(Will be updated after experiments)*

| Method | Accuracy | F1 Weighted | F1 Macro | Trainable % |
|--------|----------|-------------|----------|-------------|
| Full Fine-Tuning | — | — | — | 100% |
| LoRA | — | — | — | ~0.17% |
| Adapter | — | — | — | ~0.67% |
| Prefix Tuning | — | — | — | ~0.11% |

📈 [W&B experiment dashboard](https://wandb.ai/YOUR_USERNAME/rusentiment-peft)

---

## 🗂️ Project Structure

```
rusentiment-peft/
├── configs/          # YAML experiment configs
├── src/
│   ├── data/         # Dataset & preprocessing
│   ├── models/       # Model factory (all PEFT methods)
│   ├── training/     # Trainer wrapper & metrics
│   └── utils/        # Seed, config loader, helpers
├── notebooks/        # EDA and results analysis
├── tests/            # Unit tests (pytest)
├── train.py          # Training entry point
└── evaluate.py       # Evaluation entry point
```

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/rusentiment-peft.git
cd rusentiment-peft

# 2. Install
pip install -r requirements.txt

# 3. Add W&B key
cp .env.example .env
# edit .env and paste your key from https://wandb.ai/authorize

# 4. Download dataset from Kaggle and place CSV in data/raw/

# 5. Train
make train-lora        # LoRA
make train-full        # Full fine-tuning baseline
make train-adapter     # Adapter Tuning
make train-prefix      # Prefix Tuning
```

---

## 🧪 Tests

```bash
make test
# or: pytest tests/ -v
```

---

## 🧠 Methods Explained

### LoRA (Low-Rank Adaptation)
Freezes all original weights. Injects trainable rank-decomposition matrices **A** and **B** into attention layers. The weight update is `ΔW = BA` where `B ∈ R^{d×r}` and `A ∈ R^{r×k}` with `r << d`. Extremely parameter-efficient.

### Adapter Tuning
Inserts small feed-forward "adapter" modules after each transformer sub-layer. Only adapter parameters are trained. The bottleneck architecture compresses representations to a small hidden size before projecting back.

### Prefix Tuning
Prepends a set of learnable "virtual tokens" (prefix) to the key and value matrices of every attention layer. The original model is completely frozen; only prefix parameters are updated.

---

## 📚 References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) — Li & Liang, 2021
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) — Houlsby et al., 2019
- [HuggingFace PEFT library](https://github.com/huggingface/peft)

---

## 👤 Author

**Your Name** — [GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
