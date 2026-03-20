"""Unit tests for utility functions."""
import pytest
import torch
from src.utils.utils import set_seed, load_config, count_parameters


def test_set_seed_reproducibility():
    set_seed(42)
    a = torch.rand(5)
    set_seed(42)
    b = torch.rand(5)
    assert torch.allclose(a, b), "set_seed should produce identical tensors"


def test_load_config(tmp_path):
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("model:\n  num_labels: 3\nmethod: lora\n")
    config = load_config(str(cfg_file))
    assert config["method"] == "lora"
    assert config["model"]["num_labels"] == 3


def test_count_parameters():
    # Tiny model for testing
    model = torch.nn.Linear(10, 3)
    result = count_parameters(model)
    assert result["total"] == 33          # 10*3 weights + 3 bias
    assert result["trainable"] == 33
    assert result["trainable_pct"] == 100.0
