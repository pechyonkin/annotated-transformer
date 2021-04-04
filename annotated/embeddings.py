"""Transformer embeddings."""
import math

from torch import nn


class Embeddings(nn.Module):
    """In our model, we share the same weight matrix between the two embedding
    layers and the pre-softmax linear transformation. In the embedding layers,
    we multiply those weights by sqrt(d_model)."""

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
