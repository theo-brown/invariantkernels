from typing import Callable, Iterable, Optional, Tuple

import gpytorch
import torch


class InvariantKernel(gpytorch.kernels.Kernel):
    r"""A kernel that is invariant to a collection of transformations.

    Currently only supports group invariance.

    The invariant kernel is defined as:
    .. math::
        k_G(x, y) = \frac{1}{|G|^2} \sum_{g \in G} \sum_{h \in G} k(g(x), h(y))

    If the kernel is isotropic, we can use the simpler form:
    .. math::
        k_G(x, y) = \frac{1}{|G|} \sum_{g \in G} k(g(x), y)

    where :math:`G` is the group of transformations, :math:`k` is the base kernel, and :math:`x` and :math:`y` are the inputs.
    """

    def __init__(
        self,
        base_kernel: gpytorch.kernels.Kernel,
        transformations: Callable[[torch.tensor], torch.tensor],
        is_isotropic: bool = False,
        is_group: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.base_kernel = base_kernel
        self.transformations = transformations
        self.is_isotropic = is_isotropic
        self.is_group = is_group

        if not self.is_group:
            raise NotImplementedError("InvariantKernel only supports group invariance.")

    def forward(
        self, x1, x2, diag: bool = False, last_dim_is_batch: bool = False, **kwargs
    ) -> torch.tensor:
        if last_dim_is_batch:
            raise NotImplementedError(
                "last_dim_is_batch=True not supported for GroupInvariantKernel."
            )

        x1_orbits = self.transformations(x1)  # Shape is ... x G x N x d
        G = x1_orbits.shape[-3]
        if self.is_isotropic:
            # Sum is over a single set of orbits
            # x2_orbits should be constructed by tiling x2 along the -3 axis
            dims = [-1] * (x2.dim() + 1)
            dims[-3] = G
            x2_orbits = x2.unsqueeze(-3).expand(dims)
            K_orbits = self.base_kernel.forward(x1_orbits, x2_orbits)
            K = torch.mean(K_orbits, dim=-3)
        else:
            # Sum is over all pairs of orbits
            # x2_orbits should be constructed by applying the transformations to x2
            x2_orbits = self.transformations(x2)  # Shape is ... x G x M x d

            if x2_orbits.shape[-3] != G or x1_orbits.shape[-3] != G:
                raise ValueError(
                    "Different numbers of orbits for x1 and x2. "
                    "Check that self.transformations returns a tensor of shape (..., G, N, d)."
                )

            # WARNING: This is quadratic in the number of orbits!
            # TODO: We should be able to parallelise this
            # TODO: Should be a way of doing it with less memory
            # Repeat each element of x1_orbits G times
            # New shape is G^2 x ... x N x d
            x1_orbits_expanded = x1_orbits.repeat_interleave(G, dim=-3)
            # Repeat the entire x2_orbits G times
            # New shape is ... x G^2 x M x d
            dims = [1] * x2_orbits.dim()
            dims[-3] = G
            x2_orbits_expanded = x2_orbits.repeat(dims)
            # Compute the kernel between each pair of expanded orbits = all combinations of orbits
            K_orbits = self.base_kernel.forward(x1_orbits_expanded, x2_orbits_expanded)
            K = torch.mean(K_orbits, dim=-3)

        if diag:
            return K.diag()
        else:
            return K
