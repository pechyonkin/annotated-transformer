"""The Annotated Transformer.

2021 version, with Python type hints and tests.
"""
import copy

import torch.nn.functional as F
from torch import nn

from annotated.attention import MultiHeadedAttention
from annotated.decoder import DecoderLayer, Decoder
from annotated.embeddings import Embeddings
from annotated.encoder import Encoder, EncoderLayer
from annotated.feed_forward import PositionwiseFeedForward
from annotated.positional_encoding import PositionalEncoding


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model: int, vocab: int):
        super(Generator, self).__init__()
        # project d_model input to vocab output
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # softmax along the last dimension
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Sequential,
        tgt_embed: nn.Sequential,
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


def make_model(
    src_vocab,
    tgt_vocab,
    num_layers: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
) -> EncoderDecoder:
    """Helper: Construct a model from hyperparameters."""

    c = copy.deepcopy
    attn = MultiHeadedAttention(num_attn_heads=h, d_model=d_model)
    ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
    position = PositionalEncoding(d_model=d_model, dropout=dropout)
    encoder = Encoder(
        layer=EncoderLayer(
            size=d_model,
            self_attn=c(attn),
            feed_forward=c(ff),
            dropout=dropout,
        ),
        num_layers=num_layers,
    )
    decoder = Decoder(
        layer=DecoderLayer(
            size=d_model,
            self_attn=c(attn),
            src_attn=c(attn),
            feed_forward=c(ff),
            dropout=dropout,
        ),
        num_layers=num_layers,
    )
    src_embed = nn.Sequential(
        Embeddings(d_model=d_model, vocab=src_vocab),
        c(position),
    )
    tgt_embed = nn.Sequential(
        Embeddings(d_model=d_model, vocab=tgt_vocab),
        c(position),
    )
    generator = Generator(d_model, tgt_vocab)
    model = EncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        generator=generator,
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
