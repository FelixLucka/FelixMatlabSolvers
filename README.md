# FelixMatlabSolvers
This is a collection of MATLAB functions that are used by several of my projects. I assembled them over the years and decided to share them here. I continuously modify them according to the needs of my own projects, so do not expect them to be consistent or compatible with older versions.

## Installation

1. Download the functions and make sure to add them to MATLAB's path.

```
addpath(genpath('<path to the folder>/FelixMatlabSolvers/'))
```
2. Download the [FelixMatlabTools](https://github.com/FelixLucka/FelixMatlabTools) functions and make sure to add them to MATLAB's path.
3. Some functions make use of the algebraic multigrid solvers contained in the [iFEM software](https://github.com/lyc102/ifem): download the code and copy it's subfolder `solver` into the `external` folder.
4. Some examples showcase CT reconstruction and make use of the [ASTRA toolbox](https://astra-toolbox.com/) and the [SPOT toolbox](https://github.com/mpf/spot). If you want to run these examples, make sure to install them and add them to the path. 

## Getting started

The scripts in examples/ try to exemplify the most commonly used functions.

## Contributors

Felix Lucka
