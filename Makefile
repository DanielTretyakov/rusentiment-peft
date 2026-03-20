.PHONY: install test train-full train-lora train-adapter train-prefix eval-lora clean

# --- Setup ---
install:
	pip install -r requirements.txt

# --- Tests ---
test:
	pytest tests/ -v

# --- Training ---
train-full:
	python train.py --config configs/full_finetune.yaml

train-lora:
	python train.py --config configs/lora.yaml

train-adapter:
	python train.py --config configs/adapter.yaml

train-prefix:
	python train.py --config configs/prefix_tuning.yaml

# --- Evaluation ---
eval-lora:
	python evaluate.py --config configs/lora.yaml --checkpoint experiments/lora/

eval-full:
	python evaluate.py --config configs/full_finetune.yaml --checkpoint experiments/full_finetune/

# --- Cleanup ---
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -name "*.pyc" -delete
