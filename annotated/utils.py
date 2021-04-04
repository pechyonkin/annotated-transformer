import copy

from torch import nn


def clones(module: nn.Module, num_clones: int) -> nn.ModuleList:
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_clones)])
