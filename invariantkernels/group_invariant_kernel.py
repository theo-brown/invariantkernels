from typing import Callable

import gpytorch
import torch


class GroupInvariantKernel(gpytorch.kernels.Kernel):
    r"""A kernel that is invariant to a group of transformations.

    The group-invariant kernel is defined as

    .. math::
        k_G(x, y) = \frac{1}{|G|^2} \sum_{g \in G} \sum_{h \in G} k(g(x), h(y)),

    where :math:`G` is the group of transformations, :math:`k` is the base kernel,
    and :math:`x` and :math:`y` are the inputs.

    If the kernel is isotropic (i.e., same lengthscale for all dimensions),
    we can use the lower-cost form

    .. math::
        k_G(x, y) = \frac{1}{|G|} \sum_{g \in G} k(g(x), y).

    The *normalised* group-invariant kernel is defined as:

    .. math::
        \bar{k_G}(x, y) = \frac{k_G(x,y)}{\sqrt{k_G(x, x) k_G(y, y)}}.
    """

    def __init__(
        self,
        base_kernel: gpytorch.kernels.Kernel,
        transformations: Callable[[torch.tensor], torch.tensor],
        isotropic: bool = False,
        normalize: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.base_kernel = base_kernel
        self.transformations = transformations
        self.isotropic = isotropic
        self.normalize = normalize

    def forward(
        self,
        x1: torch.tensor,
        x2: torch.tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **kwargs,
    ) -> torch.tensor:
        # Note: use lazy evaluations throughout (ie __call__ rather than forward) to
        # reduce compute cost. This means some torch operations aren't possible
        # (e.g. .sum((-3, -4)) -> .sum(-3).sum(-3))
        if self.isotropic and not self.normalize:
            # Single sum normalised by G
            x1_orbits = self.transformations(x1)  # (..., G, N, d)
            K_x1x2 = self.base_kernel(x1_orbits, x2.unsqueeze(-3)).sum(-3)
            K = K_x1x2 / x1_orbits.shape[-3]
        elif self.isotropic and self.normalize:
            # Single sum normalised by sqrt(K_G(x,x) K_G(y, y))
            x1_orbits = self.transformations(x1)  # (..., G, N, d)
            x2_orbits = self.transformations(x2)  # (..., G, M, d)
            x1_orbits_transpose = x1_orbits.transpose(-2, -3)  # (..., N, G, d)
            x2_orbits_transpose = x2_orbits.transpose(-2, -3)  # (..., M, G, d)
            K_x1x2 = self.base_kernel(x1_orbits, x2.unsqueeze(-3)).sum(-3)
            K_x1x1 = (
                self.base_kernel(x1_orbits_transpose, x1.unsqueeze(-2)).sum(-1).sum(-1)
            )
            K_x2x2 = (
                self.base_kernel(x2_orbits_transpose, x2.unsqueeze(-2)).sum(-1).sum(-1)
            )
            normalisation = torch.sqrt(K_x1x1.unsqueeze(-1) * K_x2x2.unsqueeze(-2))
            K = K_x1x2 / normalisation
        elif not self.isotropic and not self.normalize:
            # Double sum normalised by G
            x1_orbits = self.transformations(x1)  # (..., G, N, d)
            x2_orbits = self.transformations(x2)  # (..., G, M, d)
            # Shape (..., N, M)
            K_x1x2 = (
                self.base_kernel(x1_orbits.unsqueeze(-3), x2_orbits.unsqueeze(-4))
                .sum(-3)
                .sum(-3)
            )
            K = K_x1x2 / x1_orbits.shape[-3] ** 2
        elif not self.isotropic and self.normalize:
            # Double sum normalised by sqrt(K_G(x,x) K_G(y, y))
            x1_orbits = self.transformations(x1)  # (..., G, N, d)
            x2_orbits = self.transformations(x2)  # (..., G, M, d)

            x1_orbits_transpose = x1_orbits.transpose(-2, -3)  # (..., N, G, d)
            x2_orbits_transpose = x2_orbits.transpose(-2, -3)  # (..., M, G, d)

            # Shape (..., N, M)
            K_x1x2 = (
                self.base_kernel(x1_orbits.unsqueeze(-3), x2_orbits.unsqueeze(-4))
                .sum(-3)
                .sum(-3)
            )
            # Shape (..., N)
            K_x1x1 = (
                self.base_kernel(x1_orbits_transpose, x1_orbits_transpose)
                .sum(-1)
                .sum(-1)
            )
            # Shape (..., M)
            K_x2x2 = (
                self.base_kernel(x2_orbits_transpose, x2_orbits_transpose)
                .sum(-1)
                .sum(-1)
            )
            # Shape (..., N, M)
            normalisation = torch.sqrt(K_x1x1.unsqueeze(-1) * K_x2x2.unsqueeze(-2))
            K = K_x1x2 / normalisation
        else:
            # Never raised
            raise NotImplementedError

        # TODO: ensure outputs are lazy
        if diag:
            return K.to_dense().diag()
        else:
            return K
