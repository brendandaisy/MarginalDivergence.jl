using Test
using Revise
using DEParamDistributions
using Distributions
using DelayDiffEq, OrdinaryDiffEq

## Setup
sir_pdist = SIRParamDistribution(S₀=Beta(1, 1), β=Normal(0.3, 0.01))

@testset "Simulating an ODE" begin
    sim = simulate(sir_pdist, 100; saveat=1:10)
    @test length(sim) == 100
    @test length(sim.u[1]) == 10
    sim = simulate(sir_pdist, 100; saveat=1:10, keep=true)
    @test sim[1] isa ODESolution
    @test length(sim[1].u[1]) == 2
end

@testset "Simulation with precomputed values" begin
    sim = simulate([(S₀=.05, β=0.3), (S₀=.99, β=0.4)], sir_pdist; save_idxs=1)
    @test sum(sim.u[1]) < sum(sim.u[2])
    sim = simulate([(S₀=.0, β=0.3)], sir_pdist; save_idxs=1, saveat=5:10)
    @test all(sim.u[1] .≈ 0.)
end