from typing import Callable

from torch import nn, Tensor

from annotated.norm import LayerNorm
from annotated.utils import clones

SUBLAYERS_IN_ENCODER_LAYER = 2


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer: nn.Module, num_layers: int):
        super(Encoder, self).__init__()
        self.layers: nn.ModuleList = clones(layer, num_layers)
        self.norm: nn.Module = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    Note: I think this should be name "residual connection" instead, since what
    this does is apply residual connection around the layer's output.

    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(features=size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: Callable) -> Tensor:
        """Apply residual connection to any sublayer with the same size."""

        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size: int, self_attn, feed_forward, dropout: float):
        """

        Args:
            size:
            self_attn:
            feed_forward:
            dropout: dropout rate
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(
            module=SublayerConnection(size, dropout),
            num_clones=SUBLAYERS_IN_ENCODER_LAYER,
        )
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""

        # sublayers' connection call signature: (x: Tensor, sublayer: Callable)
        # multi-head self-attention mechanism
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, mask))
        # simple, position-wise fully connected feed-forward network
        return self.sublayer[1](x, self.feed_forward)
