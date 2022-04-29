using DEParamDistributions
using Test
using Distributions
using OrdinaryDiffEq
import Turing: summarystats
using Random

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

@testset "Make sure importance sampling methods don't error" begin
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

Random.seed!(1234)
pfixed = SIRParamDistribution(S₀=0.6, β=1.25, α=0.2)
pdist = SIRParamDistribution(S₀=Uniform(0.1, 0.9), β=Uniform(0.3, 3.), α=Uniform(0.05, 0.3))
dekwargs = (saveat=1, save_idxs=2)
true_sim = solve_de_problem(pdist, (S₀=0.6, β=1.25, α=0.2); dekwargs...).u
cond_sims = simulate((S₀=0.6,), pdist, 1000; dekwargs...)

# true_obs_mod = PoissonBiasMult(1000., 3.)
# obs_mod = PoissonBiasMult(1000., truncated(Gamma(2, 1), 1, Inf))
# y = rand(obs_tspan(true_sim, true_obs_mod, 15))
obs_mod = PoissonTests(1000.)
om_samps = [sample_obs_mod(obs_mod) for _=1:1000];
marg_ldists = map((s, m)->obs_tspan(s, m, 15), cond_sims, om_samps);
marginal_likelihood(y, marg_ldists)
