```julia
using MarginalDivergence
using Distributions, MonteCarloMeasurements
using Parameters
```




### Define the latent process

There are two classes of interfaces: models inheriting from `GenericModel` must simply define a `solve` method, while
models inheriting from `ODEModel` and `DDEModel`equires implementing at least `timespan` `initial_values` `parameters` and `de_func`, 
which point to the relavent componenets of DifferentialEquations.jl's implementation. 
The `solve` method as well as some other helpers are then inherited.

Here we implement a simple `ODEModel`, the SIS model. First, we define the differential equations:

```julia
function sis!(dx, x, p, t)
    I = x[1]
    β, α = p
    dx[1] = β * I * (1-I) - α * I
end
```

```
sis! (generic function with 1 method)
```





These should be implemented in the same way as in `DifferentialEquations.jl`. Now we implement our model object,
which stores the parameters, timespan, etc, as well as complete the `ODEModel` interface.

```julia
@with_kw struct SISModel{T<:Real} <: ODEModel{T}
    start::T = 0.
    stop::T = 20.
    I₀::Param{T} = 0.01
    β::Param{T} = 0.3
    α::Param{T} = 0.1
end

MarginalDivergence.timespan(m::SISModel) = (m.start, m.stop)
MarginalDivergence.initial_values(m::SISModel) = [m.I₀]
MarginalDivergence.parameters(m::SISModel) = [m.β, m.α]
MarginalDivergence.de_func(::SISModel) = sis!
```



And we're all set. Define an instance of the model with uncertain $\beta$ and $\alpha$, seeing data only up to $T=10$.

```julia
mod = SISModel(β=0..1, α=0.1..0.5)
sol = solve(mod; saveat=0:10, save_idxs=1).u
```

```
11-element Vector{Particles{Float64, 2000}}:
 0.01
 0.0127 ± 0.0039
 0.0175 ± 0.011
 0.0258 ± 0.023
 0.0393 ± 0.044
 0.0594 ± 0.077
 0.0858 ± 0.12
 0.116 ± 0.16
 0.146 ± 0.2
 0.174 ± 0.23
 0.198 ± 0.25
```





To compute the RMD of $\beta$, we also simulate the model holding $\beta$ fixed to the "true" value.

```julia
θtrue = (β=0.7, α=0.2)
sol_restr = solve(mod, (β=0.7,); saveat=0:10, save_idxs=1).u
```

```
11-element Vector{Particles{Float64, 2000}}:
 0.01
 0.0149 ± 0.0017
 0.0224 ± 0.0051
 0.0337 ± 0.011
 0.0508 ± 0.022
 0.0758 ± 0.039
 0.111 ± 0.065
 0.156 ± 0.098
 0.208 ± 0.13
 0.265 ± 0.17
 0.319 ± 0.2
```





Finally, we need to define the observation process, here using the provided Gaussian noise model, and get the Distribution
of the data.

```julia
obs_mod = NConstVar(0.01)

sol_true = solve(mod, θtrue; saveat=0:10, save_idxs=1).u
ytrue = observe_dist(obs_mod, sol_true)
```

```
11-element Vector{Normal{Float64}}:
 Normal{Float64}(μ=0.01, σ=0.1)
 Normal{Float64}(μ=0.016338818919136543, σ=0.1)
 Normal{Float64}(μ=0.02654426937177269, σ=0.1)
 Normal{Float64}(μ=0.04273385817344045, σ=0.1)
 Normal{Float64}(μ=0.06782393946706626, σ=0.1)
 Normal{Float64}(μ=0.10533481065432604, σ=0.1)
 Normal{Float64}(μ=0.15850378577161595, σ=0.1)
 Normal{Float64}(μ=0.22844561737900243, σ=0.1)
 Normal{Float64}(μ=0.31192175591246146, σ=0.1)
 Normal{Float64}(μ=0.40075126698866187, σ=0.1)
 Normal{Float64}(μ=0.48437783675404145, σ=0.1)
```





And we get the RMD, approximating with 100 outer product steps (in practice, $N$ and $M$ should probably be much larger)
```julia
marginal_divergence(ytrue, sol_restr, sol, obs_mod; N=100)
```

```
0.8384965917102682
```


