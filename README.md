# DEParamDistributions.jl

WIP Convenience package for integrating differential equations and random variables. Features Bayesian inference using Turing.jl and importance sampling, Monte Carlo simulation using SciML's EnsembleAnalysis, and an interface for safe and easy access to initial values, parameters, etc. 

## Installation

Only tested using Julia v1.7.1. Using Pkg mode, simply

```
add https://github.com/brendandaisy/DEParamDistributions.jl
```

to create a static installation or use `dev` to clone a version you can edit.

Check the package is loaded by typing `st` in Pkg model. To test everything is working:

```
test DEParamDistributions
```

To update to the latest version, simply
```
update DEParamDistributions
```

## Examples

A number of examples can be found in the test folder.