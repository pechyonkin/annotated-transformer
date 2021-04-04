import numpy as np
import torch


def test_subsequent_mask() -> None:
    n = np.random.randn(5, 6, 7)
    _ = torch.from_numpy(n)
