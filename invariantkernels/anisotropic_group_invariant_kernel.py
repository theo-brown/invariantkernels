from typing import Callable, Iterator

import gpytorch
import torch


class AnisotropicGroupInvariantKernel(gpytorch.kernels.Kernel):
    r"""Construct a kernel that is invariant to a group of transformations.

    The *anisotropic* invariant kernel is given by:
    .. math::
        \begin{equation*}
            K_G(x, x') = \frac{1}{|G|^2} \sum_{\tau \in G} \sum_{\sigma \in G} k(\tau(x), \sigma(x'))
        \end{equation*}

    Note that this works for both isotropic and anisotropic kernels. However,
    isotropic kernels can be computed more efficiently using the
    `IsotropicGroupInvariantKernel` class.

    Parameters
    ----------
    base_kernel : gpytorch.kernels.Kernel
        The kernel to modify.

    transformation_group : Callable[torch.Tensor, Iterator[torch.Tensor]]
        A function that takes an x and returns a generator over all transformed versions of x (i.e., the orbits of x).
    """

    def __init__(
        self,
        base_kernel: gpytorch.kernels.Kernel,
        transformation_group: Callable[[torch.tensor], Iterator[torch.tensor]],
        data_loader_batch_size: int = 1,
        data_loader_num_workers: int = 0,
        **kwargs,
    ) -> None:
        if base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims

        super().__init__(**kwargs)

        self.base_kernel = base_kernel
        self.transformation_group = transformation_group
        self.data_loader_batch_size = data_loader_batch_size
        self.data_loader_num_workers = data_loader_num_workers

    def forward(
        self, x1, x2, diag: bool = False, last_dim_is_batch: bool = False, **kwargs
    ) -> torch.tensor:
        # TODO: Use __call__, which means we need to make it lazy

        if last_dim_is_batch:
            raise NotImplementedError(
                "last_dim_is_batch=True not implemented for AnisotropicGroupInvariantKernel."
            )

        # Note: we are optimising for memory usage first, and then performance afterwards.
        N = x1.shape[-2]
        M = x2.shape[-2]
        if len(x1.shape) == 2:
            K = torch.zeros(N, M, device=x1.device, dtype=x1.dtype)
        elif len(x1.shape) == 3:
            K = torch.zeros(x1.shape[0], N, M, device=x1.device, dtype=x1.dtype)
        else:
            raise ValueError("Input tensor must have either 2 or 3 dimensions.")

        n_transforms = 0
        for transformed_x1 in self.transformation_group(x1):
            for transformed_x2 in self.transformation_group(x2):
                K += self.base_kernel.forward(transformed_x1, transformed_x2).squeeze()
                n_transforms += 1

        K /= n_transforms

        if diag:
            # TODO: Implement diagonal computation that doesn't require storing the full kernel matrix
            return K.diag()
        else:
            return K
