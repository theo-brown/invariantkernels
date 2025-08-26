from typing import Callable

import gpytorch
import h5py
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from tqdm import tqdm

from invariantkernels import GroupInvariantKernel, permutation_group


def create_synthetic_objective(
    d: int, kernel: gpytorch.kernels.Kernel, seed: int, n: int, jitter: float = 1e-6
) -> Callable[torch.tensor, torch.tensor]:
    """Create a synthetic objective function, which is the mean of a GP with the given kernel."""
    torch.manual_seed(seed)

    # Generate samples from a random function
    x = torch.rand(n, d)
    mean = gpytorch.means.ZeroMean()
    prior = gpytorch.distributions.MultivariateNormal(mean(x), kernel(x))
    y = prior.sample()

    # Fit a GP to the samples
    true_gp = SingleTaskGP(
        x,
        y.unsqueeze(-1),
        jitter * torch.ones_like(y.unsqueeze(-1)),
        covar_module=kernel,
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(true_gp.likelihood, true_gp)
    fit_gpytorch_mll(mll)

    # Use the mean of that GP as our objective function
    def f(x):
        return true_gp(x).mean.detach()

    return f


def save_to_file(f: str, d: dict[str, torch.tensor]) -> None:
    """Save a dict of tensors to HDF5."""
    with h5py.File(f, "a") as h5:
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                h5[k] = v.detach().cpu().numpy()
            else:

                h5[k] = v


def run(
    seed: int,
    n_bayesopt_steps: int,
    d: int,
    objective_seed: int,
    output_file: str,
    output_group: str,
    objective_lengthscale,
    transformations: Callable[torch.tensor, torch.tensor],
    n_objective_initialisation_points: int,
    noise_variance: float,
    use_normalized_objective_kernel: bool,
    use_invariant_kernel: bool,
    use_normalized_kernel: bool,
) -> None:
    """Run BayesOpt on a synthetic objective."""
    # Generate objective function
    print("Generating objective...")
    bounds = torch.tensor([[0.0, 1.0] for _ in range(d)]).T
    base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
    base_kernel.lengthscale = torch.tensor([objective_lengthscale], requires_grad=False)
    objective_kernel = GroupInvariantKernel(
        base_kernel=base_kernel,
        transformations=transformations,
        normalize=use_normalized_objective_kernel,
    )

    f = create_synthetic_objective(
        d=d,
        kernel=objective_kernel,
        seed=objective_seed,
        n=n_objective_initialisation_points,
    )

    def f_noisy(x):
        return f(x) + noise_variance * torch.randn(1)

    save_to_file(
        output_file,
        {
            f"{output_group}/objective/seed": objective_seed,
            f"{output_group}/objective/n_initial_points": n_objective_initialisation_points,
            f"{output_group}/objective/lengthscale": objective_lengthscale,
            f"{output_group}/objective/normalized": use_normalized_objective_kernel,
        },
    )
    print("Done.")

    # Reseed the RNG with the experiment seed
    torch.manual_seed(seed)

    # Initial observation
    print("Initial sampling...")
    train_x = torch.rand(1, d)
    train_y = f_noisy(train_x)
    print("Done.")

    # Create arrays to store reported values in
    reported_x = torch.empty((n_bayesopt_steps, d))
    reported_f = torch.empty(n_bayesopt_steps)

    # Create kernel
    if use_invariant_kernel:
        kernel = GroupInvariantKernel(
            base_kernel=gpytorch.kernels.MaternKernel(nu=2.5),
            transformations=transformations,
            normalize=use_normalized_kernel,
        )
    else:
        kernel = gpytorch.kernels.MaternKernel(nu=2.5)

    print("Running BO...")
    for i in tqdm(range(n_bayesopt_steps)):
        # Refit the hyperparameters of the GP
        model = SingleTaskGP(
            train_x,
            train_y.unsqueeze(-1),
            covar_module=kernel,
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Maximise the acquisition function
        next_x, _ = optimize_acqf(
            UpperConfidenceBound(model, beta=2.0),
            bounds,
            q=1,
            num_restarts=8,
            raw_samples=1024,
        )
        # Make observation
        next_y = f_noisy(next_x)
        next_f = f(next_x)

        # Update training data
        train_x = torch.cat([train_x, next_x])
        train_y = torch.cat([train_y, next_y])
        reported_x[i] = next_x.squeeze()
        reported_f[i] = next_f

    print("Done.")

    # Save to file
    print("Saving...")
    save_to_file(
        output_file,
        {
            f"{output_group}/observed_x": train_x,
            f"{output_group}/observed_y": train_y,
            f"{output_group}/reported_x": reported_x,
            f"{output_group}/reported_f": reported_f,
        },
    )
    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a synthetic BayesOpt experiment with invariances."
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device to run on (e.g., 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed for the experiment."
    )
    parser.add_argument(
        "--n_bayesopt_steps",
        type=int,
        required=True,
        help="Number of Bayesian optimization steps.",
    )
    parser.add_argument(
        "-d", type=int, required=True, help="Dimensionality of the problem."
    )
    parser.add_argument(
        "--objective_seed",
        type=int,
        required=True,
        help="Random seed for the objective function.",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="File path to save the output."
    )
    parser.add_argument(
        "--output_group",
        type=str,
        required=True,
        help="Group name for organizing output (e.g., in HDF5).",
    )
    parser.add_argument(
        "--objective_lengthscale",
        type=float,
        default=0.1,
        help="Lengthscale for the objective function's kernel.",
    )
    parser.add_argument(
        "--transformations",
        type=str,
        default="permutation_group",
        choices=["permutation_group", "cyclic_group"],
        help="Group of transformations to use for the invariance.",
    )

    parser.add_argument(
        "--n_objective_initialisation_points",
        type=int,
        default=25,
        help="Number of initial points for the objective function.",
    )
    parser.add_argument(
        "--noise_variance",
        type=float,
        default=1e-3,
        help="Variance of the observation noise.",
    )
    parser.add_argument(
        "--use_invariant_kernel",
        dest="use_invariant_kernel",
        action="store_true",
        help="Use the invariant kernel.",
    )
    parser.add_argument(
        "--use_normalized_kernel",
        dest="use_normalized_kernel",
        action="store_true",
        help="Use the normalized invariant kernel.",
    )
    parser.add_argument(
        "--use_normalized_objective_kernel",
        action="store_true",
        help="Use a normalized kernel for the objective function.",
    )

    args = parser.parse_args()

    torch.set_default_device(args.device)
    torch.set_default_dtype(torch.float64)
    print(f"PyTorch default device set to: {args.device}")
    print("PyTorch default dtype set to: torch.float64")

    if args.transformations == "permutation_group":
        transformations = permutation_group
    else:
        raise NotImplementedError(f"Unrecognised transformations {transformations}")

    run(
        args.seed,
        args.n_bayesopt_steps,
        args.d,
        args.objective_seed,
        args.output_file,
        args.output_group,
        args.objective_lengthscale,
        transformations,
        args.n_objective_initialisation_points,
        args.noise_variance,
        args.use_normalized_objective_kernel,
        args.use_invariant_kernel,
        args.use_normalized_kernel,
    )
