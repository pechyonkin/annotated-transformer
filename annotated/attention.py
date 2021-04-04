import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def attention(
    query: Tensor, key: Tensor, value: Tensor, mask=None, dropout=None
) -> Tuple[Tensor, Tensor]:
    """Compute 'Scaled Dot Product Attention'"""

    d_k = query.size(-1)  # returns last dimension of the tensor
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)  # see sketch of how this is done
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
