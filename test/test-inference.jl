using DEParamDistributions
using Test
using Distributions
using OrdinaryDiffEq
import Turing: summarystats

## Setup
sir_pdist = SIRParamDistribution(S₀=Beta(4, 1), β=TruncatedNormal(0.3, 0.5, 0, 2), α=0.15)

@testset "Premade likelihood functions" begin
    @test joint_prior(sir_pdist) isa MultivariateDistribution
    @test length(joint_prior(sir_pdist)) == 2
    @test rand.(array_binom([2, 3], [-.1, 1.])) == [0, 3]
    @test length(array_binom(1, [.5, .4])) == 2
    @test rand(joint_poisson(0, [1/2, 3/4])) == [0., 0.]
end

@testset "Prior prediction" begin
    sir_pdist = SIRParamDistribution(S₀=100., β=Normal(0.3, 0.01))
    sim = simulate(sir_pdist, 100; saveat=1:10, save_idxs=2)
    ŷ = prior_predict(sim)
    @test size(ŷ) == (100,) 
    sim = simulate(SIRParamDistribution(β=Normal(0.3, 0.01)), 100; saveat=1:10, save_idxs=2, keep=false)
    ŷ = prior_predict(sim, s->array_binom(5, s); arraylik=true)
    @test size(ŷ) == (100,)
    @test all(ŷ[3] .≥ 0) & all(ŷ[90] .≤ 5)
end

@testset "Specifying a likelihood for multiple compartments" begin
    mylik(u) = [product_distribution(Dirac.(x)) for x ∈ u]
    sol = solve(de_problem(SIRParamDistribution; saveat=1), Tsit5())
    @test length(mylik(sol.u)) == length(sol.u)
    @test mylik(sol.u)[1] isa MultivariateDistribution
    @test prior_predict(sol, mylik; arraylik=true) == sol.u
    
    sim = simulate(sir_pdist, 100; saveat=1:10)
    @test length(prior_predict(sim, mylik; arraylik=true)) == 100
end

@testset "Make sure MCMC methods don't error" begin
    prob_true = de_problem(SIRParamDistribution(S₀=100., β=0.5); saveat=7, save_idxs=2)
    pri_pdist = SIRParamDistribution(S₀=100., β=TruncatedNormal(0.3, 0.3, 0., 2.))
    inf_true = solve(prob_true, Tsit5()).u
    obs = rand(joint_poisson(inf_true))
    fit = sample_mcmc(obs, pri_pdist, joint_poisson; prob_true.kwargs..., arraylik=false)
    @test all(summarystats(fit).nt.rhat .< 1.1)
    mcmc_mean(fit)
    postθ = extract_vars(fit, pri_pdist)
    @test keys.(postθ) == fill((:β,), length(postθ))
    fitchain(fit, pri_pdist)
end

@testset "Make sure importance sampling don't error" begin
    rvs = random_vars(sir_pdist)
    pri_draws = [NamedTuple{keys(rvs)}(rand.(values(rvs))) for _=1:100]
    sim = simulate(pri_draws, sir_pdist; saveat=2, save_idxs=2)
    y = rand(joint_poisson(sim[1]))
    likdists = [joint_poisson(u) for u ∈ sim]
    W = importance_weights(y, likdists)
    μ = importance_mean(W, pri_draws)
    importance_ess(W)

    # version specifying a sampling distribution 𝐺
    gsamples = [[g1, g2] for (g1, g2) ∈ pri_draws]
    pri_dist = joint_prior(sir_pdist)
    gdist = joint_prior(sir_pdist)
    W2 = importance_weights(y, likdists; gsamples, pri_dist, gdist)
    @test all(W .≈ W2)
    μ2 = importance_mean(W2, gsamples)
    @test all(values(μ) .≈ μ2)
end