# Examples of the provided epidemic models

```julia
using Distributions
using Plots
using DEParamDistributions
```




## SIR Model

```julia
sirmod = SIRParamDistrbution(S₀=0.8)
sol = solve_de_problem(sirmod)
plot(sol)
```

```
Error: UndefVarError: SIRParamDistrbution not defined
```





## SEIR Model

```julia
seir = SEIRParamDistrbution(S₀=0.8)
seir_with_μ = SEIRParamDistrbution(S₀=0.8, μ=0.1)
sol1 = solve_de_problem(seir)
sol2 = solve_de_problem(seir_with_μ)
plot(sol1)
plot!(sol2; vars=3, lab="I (with μ=1/10)")
```

```
Error: UndefVarError: SEIRParamDistrbution not defined
```


