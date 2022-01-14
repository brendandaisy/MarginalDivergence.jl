using Revise
using Test, DEParamDistributions
using Distributions
using OrdinaryDiffEq
using Turing

@testset "SIR's AbstractODEParamDistribution Interface" begin
    sir_pdist = SIRParamDistribution(30., Beta(1, 1), Normal(0.3, 0.01), 0.1)
    psamp = DEParamDistributions.param_sample(sir_pdist)

    @test sir_pdist.pop_size == 1.
    @test all(keys(psamp) .== fieldnames(typeof(sir_pdist)))
    @test psamp.start == 0. && psamp.pop_size == 1.
    @test timespan(sir_pdist) == (0., 30.)
    @test match_initial_values(sir_pdist, psamp) == [psamp[i] for i ∈ 3:4]
    @test match_parameters(sir_pdist, psamp) == [psamp[i] for i ∈ 5:7]
end

# @testset "Matching ODE Values" begin
#     sir_pdist = SIRParamDistribution(30., Beta(1, 1), Normal(0.3, 0.01), 0.1)
#     nt = (inf_rate = 0.2, rec_init = 0.2)
#     rvs = random_vars(sir_pdist)
#     rv_samp = NamedTuple{keys(rvs)}(rand.(values(rvs)))
#     psamp = param_sample(sir_pdist)
#     @test match_initial_values(sir_pdist, nt) == [0.01, 0.2]
#     @test match_parameters(sir_pdist, nt) == [0.2, 0.1, 1.]
#     @test keys(rv_samp) == (:rec_init, :inf_rate)
#     @test eltype(rv_samp)<:Float64
#     @test match_initial_values(sir_pdist, rv_samp) == [0.01, rv_samp[1]]
#     @test match_parameters(sir_pdist, psamp) == [psamp[5], 0.1, 1.]
# end
 
@testset "Constructing ODE Problems" begin
    sir_pdist = SIRParamDistribution(30., Beta(1, 1), Normal(0.3, 0.01), 0.1)
    blank_sir = ode_problem(SIRParamDistribution)
    @test blank_sir isa ODEProblem

    sir_prob = sample_ode_problem(sir_pdist; dense=false, saveat=1.)
    sample_ode_problem!(blank_sir, sir_pdist)
    @test sir_prob.u0[1] == blank_sir.u0[1] == sir_pdist.inf_init
    @test keys(sir_prob.kwargs) == (:dense, :saveat)
    sol = solve(sir_prob, Tsit5())
    @test sol.retcode == :Success
    @test sol.t == collect(0:30)

    sir_pdist = SIRParamDistribution(Normal(30.), Beta(1, 1), Normal(0.3, 0.01), 0.1)
    tmp = deepcopy(sir_prob)
    @test_throws ErrorException sample_ode_problem!(sir_prob, sir_pdist)
    @test sir_prob.tspan == tmp.tspan
end

## Prior predictive tests
sir_pdist = SIRParamDistribution(30., Beta(1, 1), Normal(0.3, 0.01), 0.1)
pp = prior_predict(sir_pdist, 100)
@test length(pp) == 100
@test size(pp[1].u) == (31,)
yhats = predict_yhat(pp, x->DEParamDistributions.joint_binom(10, x))
# scatter(yhats, lab="")

pp = prior_predict(sir_pdist, 10, save_idxs=1:2, saveat=2)
@test length(pp[1].u) == 16 && length(pp[1].u[1]) == 2
# plot(pp)



## Posterior tests

@test rand.(DEParamDistributions.joint_binom([2, 3], [-.1, 1.])) == [0, 3]
@test length(DEParamDistributions.joint_binom(1, [.5, .4])) == 2

pdist = SIRParamDistribution(30., .3, TruncatedNormal(1, .5, .15, 1), TruncatedNormal(.1, .5, 0, .2))
prob = sample_ode_problem(pdist; save_idxs=1, saveat=0:30)
inf_curve = solve(prob, Tsit5(), save_idxs=1).u

y = rand.(Binomial.(100, inf_curve))
mod = odemodel(y, pdist, x->DEParamDistributions.binom_lik(100, x))
fit = sample(mod, NUTS(500, 0.65), MCMCThreads(), 200, 2)

@test all(summarystats(fit).nt.rhat .< 1.1)
infhat = reshape(generated_quantities(mod, fit), 400)
@test length(infhat[1]) == 31
# scatter(y / 100)
# plot!(infhat, linealpha=.1, label="")

mod_tr = odemodel(y, pdist, x->DEParamDistributions.binom_lik(100, x), var_transform=θ->(θ[1] *= 2; θ))