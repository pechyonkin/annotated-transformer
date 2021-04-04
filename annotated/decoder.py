import numpy as np
import torch
from torch import nn, Tensor

from annotated.encoder import SublayerConnection
from annotated.norm import LayerNorm
from annotated.utils import clones

SUBLAYERS_IN_DECODER_LAYER = 3


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, num_layers: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below).
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout: float):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer: nn.ModuleList = clones(
            module=SublayerConnection(size, dropout),
            num_clones=SUBLAYERS_IN_DECODER_LAYER,
        )

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""

        m = memory
        # multi-head self-attention mechanism
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, tgt_mask))
        # perform multi-head attention over the output of the encoder stack
        x = self.sublayer[1](x, lambda y: self.src_attn(y, m, m, src_mask))
        # simple, position-wise fully connected feed-forward network
        return self.sublayer[2](x, self.feed_forward)


# noinspection PyTypeChecker
def subsequent_mask(size: int) -> Tensor:
    """Mask out subsequent positions.

    Returns a tensor of booleans.

    We also modify the self-attention sub-layer in the decoder stack to prevent
    positions from attending to subsequent positions. This masking, combined
    with fact that the output embeddings are offset by one position, ensures
    that the predictions for position  𝑖 can depend only on the known outputs at
    positions less than  𝑖.

    """
    attn_shape = (1, size, size)
    subseq_mask: np.ndarray = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subseq_mask) == 0  # broadcasting
