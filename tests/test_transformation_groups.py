import torch

from invariantkernels import block_permutation_group, permutation_group


def test_permutation_group():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    result = permutation_group(x)
    expected = torch.tensor(
        [
            [[1, 2, 3], [4, 5, 6]],  # Identity
            [[1, 3, 2], [4, 6, 5]],  # Swap last two
            [[2, 1, 3], [5, 4, 6]],  # Swap first two
            [[2, 3, 1], [5, 6, 4]],  # Left shift
            [[3, 1, 2], [6, 4, 5]],  # Right shift
            [[3, 2, 1], [6, 5, 4]],  # swap first and last
        ]
    )
    assert torch.allclose(result, expected)


def test_block_permutation_group():
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    result = block_permutation_group(x, 2)
    expected = torch.tensor(
        [
            [[1, 2, 3, 4], [5, 6, 7, 8]],  # Identity
            [[3, 4, 1, 2], [7, 8, 5, 6]],  # Swap blocks
        ]
    )
    assert torch.allclose(result, expected)
