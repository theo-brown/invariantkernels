import torch
from gpytorch.kernels import MaternKernel, ScaleKernel

from invariantkernels import InvariantKernel, permutation_group


def compute_manual_kernel_matrix(k, x, y, transformations):
    N = x.shape[0]
    M = y.shape[0]
    G = transformations(x).shape[-3]
    manual_k_G_matrix = torch.zeros(N, M)
    for i in range(N):
        for j in range(M):
            # Get all permutations of x[i] and y[j]
            # These are tensors of shape (G, d)
            Gx_i = transformations(x[i].unsqueeze(0)).squeeze()
            Gy_j = transformations(y[j].unsqueeze(0)).squeeze()
            # Compute the kernel for all pairs of permutations
            for x_perm in Gx_i:
                for y_perm in Gy_j:
                    manual_k_G_matrix[i, j] += (
                        k(x_perm.unsqueeze(0), y_perm.unsqueeze(0)).to_dense().item()
                    )
    manual_k_G_matrix /= G**2
    return manual_k_G_matrix


def test_isotropic_group_invariant():
    n_datapoints = 10
    dimension = 3
    k = ScaleKernel(MaternKernel(nu=2.5))
    k_G = InvariantKernel(
        base_kernel=k,
        transformations=permutation_group,
        is_isotropic=True,
        is_group=True,
    )
    torch.manual_seed(0)

    x = torch.rand([n_datapoints, dimension])
    y = torch.rand([n_datapoints, dimension])

    k_G_matrix = k_G(x, y).to_dense()
    manual_k_G_matrix = compute_manual_kernel_matrix(k, x, y, permutation_group)
    assert torch.allclose(k_G_matrix, manual_k_G_matrix, atol=1e-5)


def test_anisotropic_group_invariant():
    n_datapoints = 10
    dimension = 3
    k = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dimension))
    k_G = InvariantKernel(
        base_kernel=k,
        transformations=permutation_group,
        is_isotropic=False,
        is_group=True,
    )
    torch.manual_seed(0)

    x = torch.rand([n_datapoints, dimension])
    y = torch.rand([n_datapoints, dimension])

    k_G_matrix = k_G(x, y).to_dense()
    manual_k_G_matrix = compute_manual_kernel_matrix(k, x, y, permutation_group)
    assert torch.allclose(k_G_matrix, manual_k_G_matrix, atol=1e-5)


def test_batch_anisotropic_group_invariant():
    batch_size = 5
    n_datapoints = 10
    dimension = 3
    k = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dimension))
    k_G = InvariantKernel(
        base_kernel=k,
        transformations=permutation_group,
        is_isotropic=False,
        is_group=True,
    )
    torch.manual_seed(0)

    x = torch.rand([batch_size, n_datapoints, dimension])
    y = torch.rand([batch_size, n_datapoints, dimension])

    k_G_matrix = k_G(x, y).to_dense()
    for i in range(batch_size):
        manual_k_G_matrix = compute_manual_kernel_matrix(
            k, x[i], y[i], permutation_group
        )
        assert torch.allclose(k_G_matrix[i], manual_k_G_matrix, atol=1e-5)
