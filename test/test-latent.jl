using Test
using Revise
using MarginalDivergence
using Distributions, MonteCarloMeasurements
# using DelayDiffEq, OrdinaryDiffEq

@testset "AbstractLatentModel general interface" begin
    struct EmptyModel{T<:Real} <: ODEModel{T}
        t₀::Param{T}
        T::Param{T}
        a::Param{T}
    end

    MarginalDivergence.timespan(m::EmptyModel) = (m.t₀, m.T)
    
    mod = EmptyModel(0., 1. ± 0.1, 1.0)
    @test mod isa AbstractLatentModel{Float64}
    @test mod.T isa Param
    @test timespan(mod) isa Tuple{Float64, Particles}
    @test_throws ErrorException initial_values(mod)

    MarginalDivergence.initial_values(m::EmptyModel) = (m.a,)
    @test initial_values(mod) == (1.0,)

    @test_throws ErrorException de_problem(mod; dense=true) # no parameters or de_func have been provided
end

## Setup
sir0 = SIRModel()
sir1 = SIRModel{Float32}(S₀=0f0..1f0, β=Particles(TruncatedNormal(0.3f0, 0.1f0, 0f0, Inf32)))

@testset "Misc. Convenience Methods for Latent Models" begin
    vars = peak_random_vars(sir1)
    @test keys(vars) == (:S₀, :β)
    @test eltype(vars) <: Particles{Float32}
    @test length(peak_random_vars(sir0)) == 0
    @test allfixed(sir0) & !allfixed(sir1)

    μ = [1, 2, 3] ± 0.1
    @test particles_index(μ, 3) == [u.particles[3] for u in μ]
    @test_throws MethodError particles_index(μ, 1:3) # multiple indexing is not implemented
end

@testset "SIR implementation" begin
    @test_throws MethodError SIRModel(α="hi")
    @test_throws UndefVarError SIRModel(α=0.2, β=0.3f0) # implicit promotion has not been implemented
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
    sol2 = solve(sir1; saveat=1f0)
    @test Symbol(sol2.retcode) == :Success
    sol3 = solve(sir1, (β=Int(1), S₀=0.5))
    @test all(eltype.(sol3.u) .== Float32)
    sol4 = solve(sir1, (;); saveat=1f0)
    @test all(getindex.(sol2.u, 2)[2:end] .≈ getindex.(sol4.u, 2)[2:end])
end

# @testset "SEIR implementation" begin
#     @test_throws MethodError DSEIRModel(α="hi")
#     @test_throws UndefVarError DSEIRModel(α=0.2, β=0.3f0) # implicit promotion has not been implemented
#     @test_throws AssertionError DSEIRModel(start=500.)
#     @test sir0 isa SIRModel{Float64}
#     @test sir1.I₀ isa Param{Float32}
#     @test timespan(sir1) isa Tuple{Float32, Float32}
#     @test all(pmean(initial_values(sir1)) .≈ [0.5f0, 0.01f0])
#     @test parameters(sir0) == [0.3, 0.1]
# end

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