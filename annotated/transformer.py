"""The Annotated Transformer.

2021 version, with Python type hints and tests.
"""

import torch.nn.functional as F
from torch import nn

from annotated.encoder import Encoder


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model: int, vocab: int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)  # project d_model input to vocab output

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)  # softmax along the last dimension


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder,
        src_embed,
        tgt_embed,
        generator: Generator,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(
        self,
        src,
        tgt,
        src_mask,
        tgt_mask,
    ):
        """Take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(
        self,
        src,
        src_mask,
    ):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        memory,
        src_mask,
        tgt,
        tgt_mask,
    ):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
