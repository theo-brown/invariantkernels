import torch
from gpytorch.kernels import MaternKernel, ScaleKernel

from invariantkernels import InvariantKernel, permutation_group


def compute_manual_kernel_matrix(k, x, y, transformations):
    group_size = len(transformations(x[0]))
    manual_k_G_matrix = torch.zeros(x.shape[0], y.shape[0])
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            for x_perm in transformations(x[i]):
                for y_perm in transformations(y[j]):
                    manual_k_G_matrix[i, j] += (
                        k(x_perm.unsqueeze(0), y_perm.unsqueeze(0)).to_dense().item()
                    )
    manual_k_G_matrix /= group_size**2
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
