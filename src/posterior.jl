## Methods for posterior approximation θ∣y

using Turing

export turingode, sample_mcmc, mcmc_mean
export mle
export fitchain, extract_vars, model_evidence
export sample_importance, importance_weights, importance_weights!, importance_mean, importance_ess

## Methods for MCMC approximation using Turing

## TODO decide how to track multiple compartments and make consistent with prior-predict
@model function turingode(
        data, prior::T1, arraylik, ::Type{T2}=Float64;
        var_transform=identity, vars=random_vars(prior), saveat, save_idxs=1
) where {T1 <: AbstractODEParamDistribution, T2 <: Real}

    theta = Vector{T2}(undef, length(vars))
    for (i, val) ∈ enumerate(vars)
        theta[i] ~ NamedDist(val, keys(vars)[i]) # name of distr goes last
    end

    samp_vals = var_transform(convert.(T2, theta))
    prob = ode_problem(prior, NamedTuple(zip(keys(vars), samp_vals)); saveat, save_idxs)
    sol = solve(prob, Tsit5())
    if sol.retcode != :Success
        Turing.@addlogprob!(-Inf)
        return
    end
    data ~ arraydist(arraylik(sol.u))
    return sol.u
end

function sample_mcmc(
    data, prior::AbstractODEParamDistribution, arraylik; 
    sampler=NUTS(), iter=500, chains=4, dekwargs...
)
    model = turingode(data, prior, arraylik; dekwargs...)
    sample(model, sampler, MCMCThreads(), iter, chains)
end

mcmc_mean(fit) = summarystats(fit).nt.mean

function extract_vars(ch::Chains, pdist::AbstractODEParamDistribution)
    vars = keys(random_vars(pdist))
    tup = Turing.get(ch, [v for v in vars]) # get named tuple of axisarrays
    map(eachindex(tup[1])) do i
        NamedTuple{vars}([t[i] for t in tup])
    end
end

## Maximum likelihood

function mle(
    data, pdist::AbstractODEParamDistribution, jointlik; 
    saveat=1., save_idxs=1, vars=random_vars(pdist)
)
    # ps = param_sample(pdist)
    # pnames = keys(ps)
    # initidx = findall(x->x ∈ initial_values(pdist), pnames)
    # pidx = findall(x->x ∈ parameters(pdist), pnames)
    println("Yes")
    pnames = keys(vars)
    θinit = rand.(values(vars))
    prob0 = ode_problem(pdist, NamedTuple{pnames}(θinit); saveat, save_idxs)
    mle_func = θ->loss(data, prob0, pdist, jointlik, θ; pnames)
    DiffEqFlux.sciml_train(mle_func, θinit)
end

# pdist is for matching up ode parameters
# θ is current param values (-> induces a likelihood distribution)
# data to plug into latest likelihood distribution
function loss(
    data, prob, pdist, jointlik, θ; pnames
) # this will be the negative log likelihood
    nt = NamedTuple{pnames}(θ)
    sol = solve(prob, Tsit5(), u0=match_initial_values(pdist, nt), p=match_parameters(pdist, nt); save_idxs=1, saveat=1.)
    loglik = logpdf(jointlik(sol.u), data)
    return -loglik
end

## Methods for importance sampling

function fitchain(ch::Chains, pdist::AbstractODEParamDistribution; d=MvNormal)
    rvname = keys(random_vars(pdist))
    ii = indexin(ch.name_map.parameters, vcat(rvname...)) ## indexer to make cols match with pdist
    Distributions.fit(d, Array(ch)[:,ii]')
end

## "highest level" version, for when reusing g/sols not necessary
## TODO unclear how to deal with when turing scenario
function sample_importance(
        data, prior::AbstractODEParamDistribution; 
        lf=joint_poisson, gf=nothing, N=4000, imp_func=importance_mean, 
        saveg=true
)
    # rvs = random_vars(prior)
    # pf = product_distribution(vcat(values(rvs)...)) # prior pdf
    # if gf === nothing
    #     gdraws = map(_->NamedTuple{keys(rvs)}(rand(pf)), 1:N)
    #     gsims = prior_predict(gdraws, prior)
    # else
    #     gdraws = map(_->NamedTuple{keys(rvs)}(rand(gf)), 1:N)
    #     gsims = prior_predict(gdraws, prior)
    # end
    # W = sample_weights(data, sols, jointlik)
    # NamedTuple{keys(rvs)}(imp_func(W, gdraws))
end

logweight(y, u, jointlik::Function) = logpdf(jointlik(u), y)
logweight(y, dist::T) where T <: Distribution = logpdf(dist, y)

function logweight(y, g, gx; lf, pd::T, gd=pd) where T <: Distribution
    logpdf(lf(gx), y) + logpdf(pd, g) - logpdf(gd, g)
end

function logweight(y, g, ld, pd::T, gd=pd) where T <: Distribution
    logpdf(ld, y) + logpdf(pd, g) - logpdf(gd, g)
end

## jointlik will be called on y, which will provide a distribution to which we call 
function importance_weights(data, sols::EnsembleSolution, jointlik)
    ℓW = map(sols) do u
        logweight(data, u, jointlik)
    end
    M = maximum(ℓW)
    W = exp.(ℓW .- M)
    W / sum(W)
end

function importance_weights(
    data, gdraw, gsim::EnsembleSolution;
    lf=new_lik, pd, gd=pd
)
    ℓW = map(zip(gdraw, gsim)) do (θ, x)
        logweight(data, θ, x; lf, pd, gd)
    end
    M = maximum(ℓW)
    W = exp.(ℓW .- M)
    W / sum(W)
end

function importance_weights(data, gdraw; ldists::Vector{T}, pd::Distribution, gd=pd) where T <: Distribution
    ℓW = map(zip(gdraw, ldists)) do (θg, ld)
        logweight(data, θg, ld, pd, gd)
    end
    M = maximum(ℓW)
    W = exp.(ℓW .- M)
    W / sum(W)
end

function importance_weights(data, ldists::Vector{T}) where T <: Distribution
    ℓw = map(p->logweight(data, p), ldists)
    M = maximum(ℓw)
    w = exp.(ℓw .- M)
    w / sum(w)
end

function importance_mean(W, g)
    mapreduce((x, y)->x .* y, .+, g, W)
end

function importance_mean(W, g::Vector{T}) where T <: NamedTuple
    NamedTuple{keys(g[1])}(importance_mean(W, values.(g)))
end

importance_ess(W) = mapreduce(x->1 / x, +, W)^2 / mapreduce(x->1 / x^2, +, W)

"""
Compute the model evidence p(y | d)\n
lf is the function that constructs Distributions p(y | d, θ)
"""
function model_evidence(
    data, design, sols::EnsembleSolution;
    lf=(d, x) -> Normal(x)
)
    map(x->pdf(lf(design, x), data), sols) |> mean
end

function model_evidence(data, ldists)
    map(d->pdf(d, data), ldists) |> mean
end

## Likelihood constructors

function array_binom(n, p)
    @assert (length(n) == length(p)) || length(n) == 1 || length(p) == 1
    Binomial.(n, max.(1e-5, p))
end

joint_binom(n, p) = product_distribution(array_binom(n, p))

array_poisson(λs) = Poisson.(max.(1e-5, λs))
joint_poisson(λs) = product_distribution(array_poisson(λs))

## makes it slightly cleaner at top level
## because can wrap in same two args for binom and pois
array_poisson(n, p) = Poisson.(max.(1e-5, n .* p))
joint_poisson(n, p) = product_distribution(array_poisson(n, p))

function new_lik(t1, t2, prop, x; budget, array=false)
    n = [budget * prop, budget * (1 - prop)]
    if array
        return array_poisson(n, [x[t1], x[t2]])
    end
    joint_poisson(n, [x[t1], x[t2]])
end

new_lik(d::Tuple, x; budget=100, array=false) = new_lik(d..., x; budget, array)