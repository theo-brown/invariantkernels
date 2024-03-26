# invariantkernels
*Transformation-invariant kernels in GPyTorch*

This package provides kernels for Gaussian processes that are invariant to transformations.
The core class is `GroupInvariantKernel`, which can be composed with other GPyTorch kernels to make them invariant to a group of transformations.
Example groups are in [`invariant_kernels/transformation_groups`](./invariantkernels/transformation_groups.py).

## Build notes 
1. Update the version in `pyproject.toml`
2. Remove the old versions from `build/` and `dist/`
3. Run `python -m` build
4. Run `twine upload dist/*`