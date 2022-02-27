using Test
using Revise
using DEParamDistributions
using Distributions
using DelayDiffEq, OrdinaryDiffEq

## Setup
sir_pdist = SIRParamDistribution(S₀=Beta(1, 1), β=Normal(0.3, 0.01))

@testset "Test type hierarchy" begin
    foo(p::AbstractDEParamDistribution) = typeof(p)
    sirtype = foo(sir_pdist)
    @test sirtype == SIRParamDistribution
    @test sirtype <: ODEParamDistribution
    @test sirtype <: AbstractDEParamDistribution
end

@testset "SIRParamDistribution implementation" begin
    @test sir_pdist.I₀ == 0.01
    @test sir_pdist.α == 0.1
    @test sir_pdist.α isa Float64
    @test timespan(sir_pdist) == (0., 30.)
    @test initial_values(sir_pdist) == (:S₀, :I₀)
end

@testset "Sampling a ParamDistribution" begin
    @test random_vars(sir_pdist) == (S₀=Beta(1, 1), β=Normal(0.3, 0.01))
    psamp = param_sample(sir_pdist)
    @test keys(psamp) == (:S₀, :β)
end

@testset "Matching ODE Values" begin
    nt = (β=0.2, I₀=0.05,  S₀=0.2)
    @test match_initial_values(sir_pdist, nt) == [0.2, 0.05]
    @test match_parameters(sir_pdist, nt) == [0.2, 0.1]

    ## handling odd matches and param typing
    @test_throws MethodError match_initial_values(sir_pdist, (α=1.,))
    @test match_parameters(sir_pdist, NamedTuple{(:α,), Tuple{Any}}(1.,)) == [sir_pdist.β, 1.]
    @test_throws MethodError match_initial_values(sir_pdist, (;))
    @test match_parameters(sir_pdist, (τ=1., β=0.5)) == [0.5, 0.1]
end

@testset "Constructing ODE Problems" begin
    blank_sir = de_problem(SIRParamDistribution)
    @test blank_sir isa ODEProblem

    sir_prob = sample_de_problem(sir_pdist; dense=false, saveat=1.)
    sample_de_problem!(blank_sir, sir_pdist)
    @test sir_prob.u0[2] == blank_sir.u0[2] == sir_pdist.I₀
    @test keys(sir_prob.kwargs) == (:dense, :saveat)
    @test values(values(sir_prob.kwargs)) == (false, 1.)
    sol = solve(sir_prob, Tsit5())
    @test sol.retcode == :Success
    @test sol.t == collect(0:30)
end

@testset "Constructing DDE Problems" begin
    blank_dseir = de_problem(DSEIRParamDistribution)
    @test blank_dseir isa DDEProblem

    dde_pdist = DSEIRParamDistribution(τ=Gamma(1, 5))
    dde_prob = sample_de_problem(dde_pdist; dense=false, saveat=1.)
    sample_de_problem!(blank_dseir, dde_pdist)
    @test dde_prob.u0[2] == blank_dseir.u0[2] == dde_pdist.E₀
    @test dde_prob.p[3] > 0
    sol = solve(dde_prob, MethodOfSteps(Tsit5()))
    @test sol.retcode == :Success
    @test sol.t == collect(0:30)
end