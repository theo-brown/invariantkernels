import itertools
from typing import Iterator

import torch


def permutation_group(x: torch.Tensor) -> Iterator[torch.Tensor]:
    """Generator that produces all permutations of the last dimension of x."""
    d = x.shape[-1]
    indices = torch.arange(0, d)
    for p in itertools.permutations(indices):
        yield x[..., p].squeeze()


def block_permutation_group(x: torch.Tensor, block_size: int) -> Iterator[torch.Tensor]:
    """Generator that produces all permutations of the last dimension of x in blocks of size block_size."""
    d = x.shape[-1]
    if d % block_size != 0:
        raise ValueError(
            f"Last dimension of x must be a multiple of block size (got {d} and {block_size} respectively)."
        )
    # block_indices is a tensor of shape (block_size, d // block_size)
    # where each column is a block of indices
    # e.g. for d = 6 and block_size = 2, block_indices is
    # tensor([[0, 2, 4], [1, 3, 5]])
    block_indices = torch.tensor(
        [range(i, i + block_size) for i in range(0, d, block_size)]
    ).T

    # Applying the permutations to the last dimension of block_indices
    # will give us all possible block permutations
    # e.g. for d = 6 and block_size = 2, we get
    # tensor([[0, 2, 4], [1, 3, 5]]), tensor([[2, 0, 4], [3, 1, 5]]), ...
    # Transpose and flatten to get
    # tensor([0, 2, 4, 1, 3, 5]), tensor([2, 0, 4, 3, 1, 5]), ...
    for p in permutation_group(block_indices):
        yield x[..., p.T.flatten()].squeeze()
