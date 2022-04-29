using DEParamDistributions
using Distributions
using Test

# @testset "Local marginal utility" begin
    pfixed = SIRParamDistribution(S₀=0.7, β=0.7)
    pdist = SIRParamDistribution(S₀=Uniform(0.1,  0.9), β=TruncatedNormal(0.4, 1, 0.15, 3))
    true_sim = solve_de_problem(pdist, (S₀=0.7, β=0.7); save_idxs=2, saveat=1).u
    cond_sims = simulate((S₀=0.7, ), pdist, 100; save_idxs=2, saveat=1)
    pri_sims = simulate(pdist, 100; save_idxs=2, saveat=1)
    local_marginal_utility(true_sim, cond_sims, pri_sims, (sim, m)->obs_tspan(sim, m, 8); N=100, obs_mod=PoissonTests(1000.))
# end

pfixed = SIRParamDistribution(S₀=0.6, β=1.25, α=0.2)
pdist = SIRParamDistribution(S₀=Uniform(0.1, 0.9), β=Uniform(0.3, 3.), α=Uniform(0.05, 0.3))
dekwargs = (saveat=1, save_idxs=2)
true_sim = solve_de_problem(pdist, (S₀=0.6, β=1.25, α=0.2); dekwargs...).u
cond_sims = simulate((S₀=0.6,), pdist, 10_000; dekwargs...)
pri_sims = simulate(pdist, 10_000; dekwargs...)
local_marginal_utility(
    true_sim, cond_sims, pri_sims, (sim, m)->obs_tspan(sim, m, 20); 
    N=2500, M=3000, true_obs_mod=PoissonBiasMult(1000., 3.), obs_mod=PoissonBiasMult(1000., truncated(Gamma(2, 1), 1, Inf))
)

# pfixed = SIRParamDistribution(stop=60., S₀=0.7, β=0.7)
# pdist = SIRParamDistribution(stop=60., S₀=0.7, β=TruncatedNormal(0.4, 1, 0.15, 3))

# precomps = all_designs_precomps((stop=60., S₀=0.7, β=0.7), pdist; saveat=5, save_idxs=2)
# # d = [3, 5, 10]
# # dlik(x) = joint_poisson(d[3], [x[d[1]], x[d[2]]])

# addprocs(4)
# @everywhere using DEParamDistributions
# local_utility(10, (β=0.7,), pdist; N=20, precomps, saveat=5, save_idxs=2)