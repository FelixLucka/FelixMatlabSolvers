# FelixMatlabSolvers
This is a collection of MATLAB functions that are used by several of my projects. I assembled them over the years and decided to share them here. I continuously modify them according to the needs of my own projects, so do not expect them to be consistent or compatible with older versions.

## Installation

1. Download the functions and make sure to add them to MATLAB's path.

```
addpath(genpath('<path to the folder>/FelixMatlabTools/'))
```
2. Some functions make use of the algebraic multigrid solvers contained in the [iFEM software](https://github.com/lyc102/ifem): download the code and copy it's subfolder `solver` into the `external` folder.

## Getting started

The scripts in Examples/ try to exemplify the most commonly used functions.

## Contributors

Felix Lucka
