using Test
using Revise
using DEParamDistributions
using Distributions, MonteCarloMeasurements
using DelayDiffEq, OrdinaryDiffEq

@testset "AbstractLatentModel general interface" begin
    struct EmptyModel{T<:Real} <: AbstractLatentModel{T}
        t₀::Param{T}
        T::Param{T}
        a::Param{T}
    end

    timespan(m::EmptyModel) = (m.t₀, m.T)
    
    mod = EmptyModel(0., 1. ± 0.1, 1.0)
    @test mod isa AbstractLatentModel{Float64}
    @test mod.T isa Param
    @test timespan(mod) isa Tuple{Float64, Particles}
    @test_throws ErrorException initial_values(mod)
end

## Setup
sir0 = SIRModel()
sir1 = SIRModel{Float32}(S₀=0f0..1f0, β=Particles(TruncatedNormal(0.3f0, 0.1f0, 0f0, Inf32)))

@testset "SIR implementation" begin
    @test_throws MethodError SIRModel(α="hi")
    @test_throws UndefVarError SIRModel(α=0.2, β=0.3f0) # implicit promotion has not been implemented
    @test_throws AssertionError SIRModel(start=500.)
    @test sir0 isa SIRModel{Float64}
    @test sir1.I₀ isa Param{Float32}
    @test timespan(sir1) isa Tuple{Float32, Float32}
    @test all(pmean(initial_values(sir1)) .≈ [0.5f0, 0.01f0])
    @test parameters(sir0) == [0.3, 0.1]
end

@testset "Solving ODE Problems" begin
    prob = de_problem(sir1; dense=false, saveat=1f0)
    @test keys(prob.kwargs) == (:dense, :saveat)
    @test values(values(prob.kwargs)) == (false, 1f0)
    sol = solve(prob, Tsit5())
    @test sol.retcode == :Success
    sol2 = solve(sir1; saveat=1f0)
    @test sol2.retcode == :Success
    @test all(getindex.(sol.u, 2)[2:end] .≈ getindex.(sol2.u, 2)[2:end])
    sol3 = solve(sir1, (β=Int(1), S₀=0.5))
    @test all(eltype.(sol3.u) .== Float32)
    sol4 = solve(sir1, (;); saveat=1f0)
    @test all(getindex.(sol.u, 2)[2:end] .≈ getindex.(sol4.u, 2)[2:end])
end

# TODO implement and test other epi models

# @testset "Constructing DDE Problems" begin
#     blank_dseir = de_problem(DSEIRParamDistribution)
#     @test blank_dseir isa DDEProblem

#     dde_pdist = DSEIRParamDistribution(τ=Gamma(1, 5))
#     dde_prob = sample_de_problem(dde_pdist; dense=false, saveat=1.)
#     sample_de_problem!(blank_dseir, dde_pdist)
#     @test dde_prob.u0[2] == blank_dseir.u0[2] == dde_pdist.E₀
#     @test dde_prob.p[3] > 0
#     sol = solve(dde_prob, MethodOfSteps(Tsit5()))
#     @test sol.retcode == :Success
#     @test sol.t == collect(0:30)
# end

##

# @testset "Sampling a ParamDistribution" begin
#     @test random_vars(sir_pdist) == (S₀=Beta(1, 1), β=Normal(0.3, 0.01))
#     psamp = param_sample(sir_pdist)
#     @test keys(psamp) == (:S₀, :β)
# end

# @testset "Matching ODE Values" begin
#     nt = (β=0.2, I₀=0.05,  S₀=0.2)
#     @test match_initial_values(sir_pdist, nt) == [0.2, 0.05]
#     @test match_parameters(sir_pdist, nt) == [0.2, 0.1]

#     ## handling odd matches and param typing
#     @test_throws MethodError match_initial_values(sir_pdist, (α=1.,))
#     @test match_parameters(sir_pdist, NamedTuple{(:α,), Tuple{Any}}(1.,)) == [sir_pdist.β, 1.]
#     @test_throws MethodError match_initial_values(sir_pdist, (;))
#     @test match_parameters(sir_pdist, (τ=1., β=0.5)) == [0.5, 0.1]
# end

# @testset "Constructing ODE Problems" begin
#     blank_sir = de_problem(SIRParamDistribution)
#     @test blank_sir isa ODEProblem

#     sir_prob = sample_de_problem(sir_pdist; dense=false, saveat=1.)
#     sample_de_problem!(blank_sir, sir_pdist)
#     @test sir_prob.u0[2] == blank_sir.u0[2] == sir_pdist.I₀
#     @test keys(sir_prob.kwargs) == (:dense, :saveat)
#     @test values(values(sir_prob.kwargs)) == (false, 1.)
#     sol = solve(sir_prob, Tsit5())
#     @test sol.retcode == :Success
#     @test sol.t == collect(0:30)
# end

# @testset "Constructing DDE Problems" begin
#     blank_dseir = de_problem(DSEIRParamDistribution)
#     @test blank_dseir isa DDEProblem

#     dde_pdist = DSEIRParamDistribution(τ=Gamma(1, 5))
#     dde_prob = sample_de_problem(dde_pdist; dense=false, saveat=1.)
#     sample_de_problem!(blank_dseir, dde_pdist)
#     @test dde_prob.u0[2] == blank_dseir.u0[2] == dde_pdist.E₀
#     @test dde_prob.p[3] > 0
#     sol = solve(dde_prob, MethodOfSteps(Tsit5()))
#     @test sol.retcode == :Success
#     @test sol.t == collect(0:30)
# end