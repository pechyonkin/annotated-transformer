"""Tests of norm module."""
import numpy as np
import torch

from annotated.norm import LayerNorm


def test_layer_norm() -> None:
    layer_norm = LayerNorm(features=5)
    x = np.random.rand(3, 4, 5)
    x = torch.tensor(x)
    _ = layer_norm.forward(x)
