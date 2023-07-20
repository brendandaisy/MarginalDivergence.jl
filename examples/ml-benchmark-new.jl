# Testing accuracy and speed of Monte Carlo estimation of marginal likelihood
# TODO: I'd like this to include several available approximation methods

using Pkg
Pkg.activate(joinpath(homedir(), ".julia/dev/MarginalDivergence/test"))
using Revise
using MarginalDivergence
using MonteCarloMeasurements, Distributions
using BenchmarkTools
using CSV, DataFrames
using Plots, StatsPlots

## helper functions

# TODO: a "comperable work" version of harmonic
function marg_lik(om, μ, data)
    lp = log_likelihood(om, μ, data)
    marginal_likelihood(lp)
end

## Coins
struct Coins <: AbstractObservationModel{Nothing} end
    
function MarginalDivergence.logpdf_particles(::Coins, μ, data)
    if any((μ .< 0) .|| (μ .> 1))
        return -Inf
    end
    sum(t->t[2]*log(t[1]) + (1 - t[2])*log(1-t[1]), zip(μ, data))
end

MarginalDivergence.observe_dist(::Coins, p) = Bernoulli.(p)

# TODO also test "restricted versions" where all but one μ is held const
# (important to get right for more advanced MC)
function coins_nml(om, μtrue; N=10_000)
    ndims = length(μtrue)
    μ_pri_sys = outer_product(fill(Uniform(0, 1), ndims), N)
    data = rand.(observe_dist(om, μtrue))
    bench = @benchmark marg_lik($om, $μ_pri_sys, $data)
    # TODO time is'nt really working since seems to be switching units
    ml_reps = map(1:100) do _
        μ_pri_sys = outer_product(fill(Uniform(0, 1), ndims), N)
        μ_pri_uni = fill(Particles(N, Uniform(0, 1), systematic=false), ndims)
        (ml_sys=marg_lik(om, μ_pri_sys, data), ml_uni=marg_lik(om, μ_pri_uni, data))
    end
    ml_true = log(2.0^(-ndims))
    return (
        nsamples=N, nparams=ndims, 
        min_time=minimum(bench).time, mean_time=mean(bench).time, 
        rmse_sys=mean((ml_est - ml_true)^2 / ml_true^2 for ml_est in getindex.(ml_reps, 1)),
        rmse_uni=mean((ml_est - ml_true)^2 / ml_true^2 for ml_est in getindex.(ml_reps, 2)),
        sd_ml_sys=std(getindex.(ml_reps, 1)),
        sd_ml_uni=std(getindex.(ml_reps, 2))
    )
end

function coins_hml(om, μtrue; N=10_000)
    ndims = length(μtrue)
    hmc = HarmonicMC(N + 10_000, 10_000, 200)
    data = rand.(observe_dist(om, μtrue)) # don't resample data! Want variance of a single p(y)
    ml_reps = map(1:100) do _
        # μinit = outer_product(fill(Uniform(0, 1), ndims), hmc.nwalkers)
        # μinit = Matrix(bootstrap(μinit, hmc.nwalkers)) # to ensure an exact (even) number of walkers
        lpost_fun = μ -> log_likelihood(om, μ, data) + sum(logpdf(Uniform(0, 1), p) for p in μ)
        # marginal_likelihood(hmc, lpost_fun, vec(eachrow(μinit)))
        marginal_likelihood(hmc, lpost_fun, collect(μtrue), 0.1; imp_dist=(m, s) -> truncated(Normal(m, 0.8s), 0, 1))
    end
    ml_true = log(2.0^(-ndims))
    return (
        nsamples=N, nparams=ndims, 
        rmse_sys=mean((ml_reps .- ml_true).^2 ./ ml_true^2),
        sd_ml_sys=std(ml_reps),
    )
end

om = Coins()

# TODO: ok we're done with this example (simple MC) implement alt MC and then move to a harder model
res_nmc1 = DataFrame(coins_nml(om, range(0, 1, round(Int, t)); N=1000) for t in range(2, 20, 10))
res_nmc2 = DataFrame(coins_nml(om, range(0, 1, round(Int, t)); N=10_000) for t in range(2, 20, 10))

CSV.write("examples/ml-coins-nmc.csv", hcat(res_nmc1, res_nmc2))

res_hmc1 = DataFrame(coins_hml(om, range(0, 1, round(Int, t)); N=1000) for t in range(2, 20, 10))

## Decaying exponentials

# TODO enforce same lengths
@kwdef struct DecayExp{T<:Real} <: GenericModel{T}
    ks::Vector{<:Param{T}}
    As::Vector{<:Param{T}}
end

function MarginalDivergence.solve(m::DecayExp; ts=[0, 1])
    [sum(A * exp(-k*t) for (k, A) in zip(m.ks, m.As)) for t in ts]
end

kprior = Exponential(1.)
ktrue = 1.5

om = NConstVar(0.01)

function samp_dexp(kprior, nparams, M; outer=true)
    if outer
        ks = outer_product(fill(kprior, nparams), M)
    else
        ks = [Particles(M, kprior) for _ in 1:nparams]
    end
    lm = DecayExp(ks, fill(0.5, nparams))
    solve(lm; ts=[1, ℯ, 2ℯ])
end

function dexp_bench(om, kprior, ktrue, nparams; M=10_000)
    lm = DecayExp(fill(ktrue, nparams), fill(0.5, nparams))
    sol_true = solve(lm; ts=[1, ℯ, 2ℯ])
    data = rand.(observe_dist(om, sol_true))

    hmc = HarmonicMC(3 * (M + 10_000), 30_000, 400)
    lpost_fun = ks -> log_likelihood(om, solve(lm, (ks=ks,); ts=[1, ℯ, 2ℯ]), data) + sum(logpdf(kprior, k) for k in ks)

    ml_reps = zeros(100, 3)
    for i=1:100
        sol_out = samp_dexp(kprior, nparams, M; outer=true)
        sol_uni = samp_dexp(kprior, nparams, M; outer=false)
        ml_reps[i, 1] = marg_lik(om, sol_out, data)
        ml_reps[i, 2] = marg_lik(om, sol_uni, data)
        ml_reps[i, 3] = marginal_likelihood(hmc, lpost_fun, [ktrue], 0.1; joint_imp_dist=imp_dist_dexp)
    end
    return (
        nsamples=M, data=data, nparams=nparams,
        sd_ml_sys=abs(std(ml_reps[:,1]) / mean(ml_reps[:,1])), 
        sd_ml_uni=abs(std(ml_reps[:,2]) / mean(ml_reps[:,2])),
        sd_ml_har=abs(std(ml_reps[:,3]) / mean(ml_reps[:,3])),
        ml_sys=ml_reps[:,1], ml_uni=ml_reps[:,2], ml_har=ml_reps[:,3]
    )
end

function imp_dist_dexp(ksamps)
    product_distribution(
        [truncated(Normal(mean(ksamp), std(ksamp) / 1.5), 0, Inf) for ksamp in eachcol(ksamps)]
    )
end

function dexp_harmonic(om, kprior, ktrue, nparams; M=10_000)
    lm = DecayExp(fill(ktrue, nparams), fill(0.5, nparams))
    sol_true = solve(lm; ts=[1, ℯ, 2ℯ])
    data = rand.(observe_dist(om, sol_true))

    hmc = HarmonicMC(M + 5000, 5000, 200)
    lpost_fun = ks -> log_likelihood(om, solve(lm, (ks=ks,); ts=[1, ℯ, 2ℯ]), data) + sum(logpdf(kprior, k) for k in ks)

    ml_reps = map(1:100) do _
        marginal_likelihood(hmc, lpost_fun, [ktrue], 0.1; joint_imp_dist=imp_dist_dexp)
    end
    return (
        nsamples=M, data=data, nparams=nparams,
        sd_ml_har=abs(std(ml_reps) / mean(ml_reps)),
        ml_har=ml_reps
    )
end

dexp_reps = [dexp_bench(om, kprior, ktrue, p; M=20_000) for p in 1:2:21]

CSV.write("ml-bench-dexp.csv", DataFrame(dexp_reps))

## SIR Model

θtrue = (S₀=0.6, β=1.25, α=0.2)
θprior = (S₀=Uniform(0.1, 0.9), β=Uniform(0.3, 3), α=Uniform(0.05, 0.3))

obs_mod = PoissonRate(1000)
sir_true = SIRModel{Float32}(;θtrue...)
inf_true = solve(sir_true; saveat=1., save_idxs=2).u # 30 observations
data = rand.(observe_dist(obs_mod, inf_true))

function samp_inf(θprior, M; outer=true)
    if outer
        S₀, β, α = outer_product(vcat(θprior...), M)
    else
        S₀=Particles(M, θprior.S₀, systematic=false)
        β=Particles(M, θprior.β, systematic=false)
        α=Particles(M, θprior.α, systematic=false)
    end
    lm = SIRModel(;S₀, β, α)
    solve(lm; saveat=1, save_idxs=2).u
end

function sir_mc(om, θprior, data; M=10_000)
    ml_reps = zeros(100, 2)
    for i=1:100
        ml_reps[i, 1] = marg_lik(om, samp_inf(θprior, M; outer=true), data)
        ml_reps[i, 2] = marg_lik(om, samp_inf(θprior, M; outer=false), data)
    end
    return (
        nsamples=M, data=data,
        # ml_sys=ml_reps[:,1], ml_uni=ml_reps[:,2],
        sd_ml_sys=std(ml_reps[:,1]), sd_ml_uni=std(ml_reps[:,2])
    )
end

function imp_dist_sir(θsamps)
    S₀, β, α = eachcol(θsamps)
    product_distribution([
        truncated(Normal(mean(S₀), 0.7 * std(S₀)), 0.1, 0.99),
        truncated(Normal(mean(β), 0.7 * std(β)), 0.3, 3),
        truncated(Normal(mean(α), 0.7 * std(α)), 0.05, 0.3)
    ])
end

function lpost_fun_sir(θ, θprior, sir_mod, obs_mod, data)
    S₀, β, α = θ
    inf = solve(sir_mod, (;S₀, β, α); saveat=1, save_idxs=2).u
    log_likelihood(obs_mod, inf, data) + sum(logpdf.(values(θprior), θ))
end

function sir_harmonic(om, sir_mod, θprior, data; M=10_000)
    hmc = HarmonicMC(M + 5000, 5000, 200)
    lpost_fun = θ -> lpost_fun_sir(θ, θprior, sir_mod, om, data)
    θinit = vcat(θtrue...)
    ml_reps = map(1:100) do _
        marginal_likelihood(hmc, lpost_fun, θinit, 0.1; joint_imp_dist=imp_dist_sir)
    end
    (nsamples=M, data=data, sd_ml_har=std(ml_reps))
end

mc_reps = [sir_mc(obs_mod, θprior, data; M) for M in round.(Int, 10 .^(2:0.5:4.5))]
har_reps = [sir_harmonic(obs_mod, sir_true, θprior, data; M) for M in round.(Int, 10 .^(2:0.5:4.5))]

CSV.write("ml-bench-sir-har.csv", DataFrame(har_reps))

# using CSV, DataFrames

# reps = CSV.read("ml-bench-res.csv", DataFrame)
# plot(
#     plot(reps.M, (10^-9*reps.mean_time)/60, yaxis=("Avg. runtime (sec.)"), lab=""),
#     plot(reps.M, reps.std_res, yaxis=("Std. error", :log), lab=""),
#     xaxis=("M")
# )
# ```