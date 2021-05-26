# VISFBPIC

A package for visualising [FBPIC](https://github.com/fbpic/fbpic) results.

## Installation

PIP package development is in progress. Currently, only Linux installation instructions are provided. Requires a working [Anaconda](https://www.anaconda.com/) distribution.

- Clone this repository.
- Configure a new Anaconda environment using the supplied `visfbpic_conda_environment.yml` file (see [conda environment documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)) and activate it. Alternatively, create your own environment following the requirements below.
- In this environment, run `pip install -e .` from the repository directory to locally install the package.

## Usage

Once installation is complete, the modules are available via a Python interpreter or script. For example:

```
>> import visfbpic
>> visfbpic.animated.animated_plasma_density('<path/to/simulation/directory/>')
```

 For now, documentation is accessed via the docstrings.

## Requirements

Recommended installation is using Anaconda, which will handle dependencies and conflicts.

- Python 3.8
- Conda packages: numba, scipy, h5py, mkl, matplotlib, numpy, pandas, tqdm, palettable
- Conda Forge packages: mpi4py, openpmd-viewer
- Pip packages (install last): fbpic

## License

[MIT](https://choosealicense.com/licenses/mit/)