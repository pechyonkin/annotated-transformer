"""Layer Normalization for Transformer."""
import torch
from torch import nn, Tensor


class LayerNorm(nn.Module):
    """Construct a LayerNorm module (See citation for details)."""

    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2: Tensor = nn.Parameter(torch.ones(features))
        self.b_2: Tensor = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor):
        """Returns a tensor of same shape as input, where each value is normalized.

        Normalization of each value:
        - subtract the mean (across feature dimension)
        - divide by STD (across feature dimension)
        - multiply by learned vector (num_features)
        - divide by learned vector (num_features)
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
