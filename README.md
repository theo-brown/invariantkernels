# invariantkernels
*Transformation-invariant kernels in GPyTorch*

This package provides kernels for Gaussian processes that are invariant to group transformations, also known as symmetries.
The core class is `InvariantKernel`, which can be composed with other GPyTorch kernels to make them invariant to a group of transformations.
Example groups are in [`invariant_kernels/transformation_groups`](./invariantkernels/transformation_groups.py).

#### Citing

The package was developed for the paper *Sample-efficient Bayesian Optimisation Using Known Invariances*, NeurIPS 2024. 
The rest of the code can be found at [theo-brown/bayesopt_with_invariances](https://github.com/theo-brown/bayesopt_with_invariances).
