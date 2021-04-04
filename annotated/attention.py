import math
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from annotated.utils import clones


def attention(
    query: Tensor, key: Tensor, value: Tensor, mask=None, dropout=None
) -> Tuple[Tensor, Tensor]:
    """Compute 'Scaled Dot Product Attention'

    Implements equation (1) from the paper.

    Query, key, value are all 4D tensors of shape: (num_batches, _, num_attn_heads, d_k)

    Where d_k = d_model // num_attn_heads

    Args:
        query:
        key:
        value:
        mask:
        dropout:

    Returns:

    """

    d_k = query.size(-1)  # returns size of last dimension of the tensor
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)  # see sketch of how this is done
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_attn_heads: int, d_model: int, dropout=0.1):
        """Take in model size and number of heads."""

        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_attn_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_attn_heads
        self.num_attn_heads = num_attn_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: Tensor, key, value, mask: Optional[Tensor] = None):
        """Implements Figure 2"""

        if mask is not None:
            # Same mask applied to all h heads.
            # insert a dimension of size one at the specified position
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)  # returns size of first dimension of the tensor

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            linear(tensor)
            .view(nbatches, -1, self.num_attn_heads, self.d_k)
            .transpose(1, 2)
            for linear, tensor in zip(
                self.linears, (query, key, value)
            )  # the 4th linear not used
        ]

        # 2) Apply attention on all the projected vectors in batch.
        # self.attn is p_attn softmax probabilities returned by attention()
        x, self.attn = attention(
            query=query, key=key, value=value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.num_attn_heads * self.d_k)
        )
        return self.linears[-1](x)  # the 4th linear used here
