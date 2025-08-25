from typing import Callable

import torch
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel

from invariantkernels import GroupInvariantKernel, permutation_group

torch.set_default_dtype(torch.float64)


def compute_manual_kernel_matrix(k, x, y, transformations, normalised) -> torch.tensor:
    # Ensure sizes of tensors are correct
    # x: b x N x d
    # y: b x M x d
    d = x.shape[-1]
    assert y.shape[-1] == d
    N = x.shape[-2]
    M = y.shape[-2]
    x_batch = x.view(-1, N, d)
    y_batch = y.view(-1, M, d)
    b = max(x_batch.shape[0], y_batch.shape[0])

    manual_k_G_matrix = torch.zeros(b, N, M)

    for b_i in range(b):
        for n_i in range(N):
            xi = x_batch[b_i, n_i]  # Shape d
            G_xi = transformations(xi.unsqueeze(0)).squeeze()  # Shape G x d

            for m_i in range(M):
                yi = y_batch[b_i, m_i]  # Shape d
                G_yi = transformations(yi.unsqueeze(0)).squeeze()  # Shape G x d

                k_xy = torch.zeros(1)
                k_xx = torch.zeros(1)
                k_yy = torch.zeros(1)

                # Sum over all σx, τy pairs
                for sigma_xi in G_xi:
                    for tau_yi in G_yi:
                        k_xy += k.forward(
                            sigma_xi.unsqueeze(0), tau_yi.unsqueeze(0)
                        ).squeeze()

                # Sum over all σx, τx pairs
                for sigma_xi in G_xi:
                    for tau_xi in G_xi:
                        k_xx += k.forward(
                            sigma_xi.unsqueeze(0),
                            tau_xi.unsqueeze(0),
                        ).squeeze()

                # Sum over all σy, τy pairs
                for sigma_yi in G_yi:
                    for tau_yi in G_yi:
                        k_yy += k.forward(
                            sigma_yi.unsqueeze(0),
                            tau_yi.unsqueeze(0),
                        ).squeeze()
                if normalised:
                    manual_k_G_matrix[b_i, n_i, m_i] = k_xy / torch.sqrt(k_xx * k_yy)
                else:
                    manual_k_G_matrix[b_i, n_i, m_i] = k_xy / G_xi.shape[-2] ** 2

    return manual_k_G_matrix


def compare_with_manual_kernel_matrix(
    *,
    lengthscale: float = 1.0,
    batch_size: int | None,
    n_datapoints: int,
    dimension: int,
    isotropic: bool,
    normalised: bool,
):
    k = ScaleKernel(MaternKernel(nu=2.5, lengthscale=lengthscale))
    k_G = GroupInvariantKernel(
        base_kernel=k,
        transformations=permutation_group,
        isotropic=isotropic,
        normalised=normalised,
    )
    torch.manual_seed(0)

    if batch_size is not None:
        x = torch.rand([batch_size, n_datapoints, dimension])
        y = torch.rand([batch_size, n_datapoints, dimension])
    else:
        x = torch.rand([n_datapoints, dimension])
        y = torch.rand([n_datapoints, dimension])

    k_G_xy_matrix = k_G(x, y).to_dense()
    k_G_xx_matrix = k_G(x, x).to_dense()

    # Check shape
    if batch_size is not None:
        assert k_G_xy_matrix.shape == torch.Size(
            [batch_size, n_datapoints, n_datapoints]
        )
        assert k_G_xx_matrix.shape == torch.Size(
            [batch_size, n_datapoints, n_datapoints]
        )
    else:
        assert k_G_xy_matrix.shape == torch.Size([n_datapoints, n_datapoints])
        assert k_G_xx_matrix.shape == torch.Size([n_datapoints, n_datapoints])

    # Check symmetry
    manual_k_G_xx_matrix = compute_manual_kernel_matrix(
        k, x, x, permutation_group, normalised
    )
    assert torch.allclose(k_G_xx_matrix, manual_k_G_xx_matrix)
    assert torch.allclose(k_G_xx_matrix.transpose(-2, -1), k_G_xx_matrix)
    assert torch.allclose(manual_k_G_xx_matrix.transpose(-2, -1), manual_k_G_xx_matrix)

    # Check values
    manual_k_G_xy_matrix = compute_manual_kernel_matrix(
        k, x, y, permutation_group, normalised
    )
    assert torch.allclose(k_G_xy_matrix, manual_k_G_xy_matrix)


def test_anisotropic_group_invariant():
    compare_with_manual_kernel_matrix(
        lengthscale=torch.tensor([1.0, 0.5, 0.1]),
        batch_size=None,
        n_datapoints=10,
        dimension=3,
        isotropic=False,
        normalised=False,
    )


def test_isotropic_group_invariant():
    compare_with_manual_kernel_matrix(
        lengthscale=1.0,
        batch_size=None,
        n_datapoints=10,
        dimension=3,
        isotropic=True,
        normalised=False,
    )


def test_anisotropic_normalised_group_invariant():
    compare_with_manual_kernel_matrix(
        lengthscale=torch.tensor([1.0, 0.5, 0.1]),
        batch_size=None,
        n_datapoints=10,
        dimension=3,
        isotropic=False,
        normalised=True,
    )


def test_isotropic_normalised_group_invariant():
    compare_with_manual_kernel_matrix(
        lengthscale=1.0,
        batch_size=None,
        n_datapoints=10,
        dimension=3,
        isotropic=True,
        normalised=True,
    )


def test_batch_anisotropic_group_invariant():
    compare_with_manual_kernel_matrix(
        lengthscale=torch.tensor([1.0, 0.5, 0.1]),
        batch_size=2,
        n_datapoints=10,
        dimension=3,
        isotropic=False,
        normalised=False,
    )


def test_batch_isotropic_group_invariant():
    compare_with_manual_kernel_matrix(
        lengthscale=1.0,
        batch_size=2,
        n_datapoints=10,
        dimension=3,
        isotropic=True,
        normalised=False,
    )


def test_batch_anisotropic_normalised_group_invariant():
    compare_with_manual_kernel_matrix(
        lengthscale=torch.tensor([1.0, 0.5, 0.1]),
        batch_size=2,
        n_datapoints=10,
        dimension=3,
        isotropic=False,
        normalised=True,
    )


def test_batch_isotropic_normalised_group_invariant():
    compare_with_manual_kernel_matrix(
        lengthscale=1.0,
        batch_size=2,
        n_datapoints=10,
        dimension=3,
        isotropic=True,
        normalised=True,
    )
