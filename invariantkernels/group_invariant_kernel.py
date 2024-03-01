from typing import Callable, Iterable

import gpytorch
import torch


class GroupInvariantKernel(gpytorch.kernels.Kernel):
    r"""Compose with an existing kernel to make it invariant to a group of transformations.

    The invariant kernel is given by:
    .. math::
        \begin{equation*}
            K_G = \sum_{\sigma \in G} k(\sigma(x), x)
        \end{equation*}


    Parameters
    ----------
    base_kernel : gpytorch.kernels.Kernel
        The kernel to modify.

    transformation_group : Callable[[torch.tensor], torch.tensor]
        A function that generates all transformed versions of an input x for a given group (i.e., the orbits of x).
        The function should take a tensor of shape (n, d) and return a tensor of shape (G, n, d) where G is the number of
        elements of the group.
    """

    def __init__(
        self,
        base_kernel: gpytorch.kernels.Kernel,
        transformation_group: Callable[[torch.tensor], Iterable[torch.tensor]],
        **kwargs,
    ) -> None:
        if base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims

        super(GroupInvariantKernel, self).__init__(**kwargs)

        self.base_kernel = base_kernel
        self.transformation_group = transformation_group

    # TODO: Not sure if this is necessary
    @property
    def is_stationary(self) -> bool:
        return self.base_kernel.is_stationary

    # TODO: This errors with "AttributeError: 'GroupInvariantKernel' object has no attribute 'has_lengthscale'"
    # @property
    # def has_lengthscale(self) -> bool:
    #     return self.base_kernel.has_lengthscale

    def forward(self, x1, x2, **kwargs) -> torch.tensor:
        x1_orbits = self.transformation_group(x1)  # Shape is G x n x d
        # Averge across the first dimension (the orbits)
        # Use `forward` instead of `__call__` to avoid the post-processing
        # steps that make the kernel a lazy object, which doesn't support torch.mean
        return torch.mean(self.base_kernel.forward(x1_orbits, x2), dim=0)

    # TODO: Not sure if this is necessary
    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

    # TODO: Not sure if this is necessary
    def prediction_strategy(
        self, train_inputs, train_prior_dist, train_labels, likelihood
    ):
        return self.base_kernel.prediction_strategy(
            train_inputs, train_prior_dist, train_labels, likelihood
        )
