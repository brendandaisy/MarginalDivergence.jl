## Methods for posterior approximation θ∣y

using Turing

export prior_predict
export turingode, sample_mcmc, mcmc_mean, fitchain, extract_vars
export importance_weights, importance_weights!, importance_mean, importance_ess
export model_evidence
export joint_prior, array_poisson, joint_poisson, array_binom, joint_binom

#= Prior Prediction =#

"""
Draw a distribution of observations from an `EnsembleSolution` or single DE solution by calling `likelihood` on each solution.

This function can return a joint or array of distributions.
"""
function prior_predict(sim::EnsembleSolution, likelihood=joint_poisson; arraylik=false)
    curves = sim
    if sim[1] isa SciMLSolution # unpack the simulations
        curves = [sol.u for sol ∈ sim]
    end
    map(curves) do c
        arraylik ? rand.(likelihood(c)) : rand(likelihood(c))
    end
end

function prior_predict(sol::SciMLSolution, likelihood=joint_poisson; arraylik=false)
    arraylik ? rand.(likelihood(sol.u)) : rand(likelihood(sol.u))
end

#= MCMC approximation using Turing =#

@model function turingode(
        data, prob, pdist::T1, likelihood=array_poisson, ::Type{T2}=Float64;
        var_transform=identity, vars=random_vars(pdist), arraylik=true
) where {T1 <: AbstractDEParamDistribution, T2 <: Real}

    theta = Vector{T2}(undef, length(vars))
    for (i, val) ∈ enumerate(vars)
        theta[i] ~ NamedDist(val, keys(vars)[i]) # name of distr goes last
    end

    samp_vals = var_transform(convert.(T2, theta))
    pr = remake_prob(prob, pdist, NamedTuple(zip(keys(vars), samp_vals)))
    sol = solve(pr, Tsit5())
    if sol.retcode != :Success
        Turing.@addlogprob!(-Inf)
        return
    end
    arraylik ? data ~ arraydist(likelihood(sol.u)) : data ~ likelihood(sol.u)
    return sol.u
end

"""
Create a Turing model and perform sampling using algorithm `sampler`.

Sampling uses `MCMCThreads` and returns a `Chains` object.
"""
function sample_mcmc(
    data, pdist::AbstractDEParamDistribution, likelihood=array_poisson; 
    sampler=NUTS(), iter=500, chains=4, arraylik=true, dekwargs...
)
    prob₀ = sample_de_problem(pdist; dekwargs...)
    model = turingode(data, prob₀, pdist, likelihood; arraylik)
    sample(model, sampler, MCMCThreads(), iter, chains)
end

mcmc_mean(fit) = summarystats(fit).nt.mean

"""
Return a vector of `NamedTuple` of posterior draws from `ch`.
"""
function extract_vars(ch::Chains, pdist::AbstractDEParamDistribution)
    vars = keys(random_vars(pdist))
    tup = Turing.get(ch, [v for v in vars]) # get named tuple of axisarrays
    map(eachindex(tup[1])) do i
        NamedTuple{vars}([t[i] for t in tup])
    end
end

"""
Approximate posterior draws from `ch` using `Distributions.fit`.

Dimensions of the resulting distribution are ordered to "match" `pdist`.
"""
function fitchain(ch::Chains, pdist::AbstractDEParamDistribution; d=MvNormal)
    rvname = keys(random_vars(pdist))
    ii = indexin(ch.name_map.parameters, vcat(rvname...)) ## indexer to make cols match with pdist
    Distributions.fit(d, Array(ch)[:,ii]')
end

#= Methods for importance sampling =#

# _logweight(y, u, jointlik::Function) = logpdf(jointlik(u), y)
_logweight(y, dist) = @inbounds logpdf(dist, y)

# function _logweight(y, g, gx; lf, pd, gd=pd)
#     logpdf(lf(gx), y) + logpdf(pd, g) - logpdf(gd, g)
# end

function _logweight(y, g, ld, pd, gd)
    @inbounds logpdf(ld, y) + logpdf(pd, g) - logpdf(gd, g)
end

"""
Compute numerically stable, normalized importance weights given a sampling distribution 𝐺(𝜃)

If any keywords missing, assume 𝐺(𝜃) = 𝑃(𝜃) and use the simplified formula
"""
function importance_weights(
    data, likdists::Vector{T}; 
    gsamples=nothing, pri_dist=nothing, gdist=nothing
) where T <: Distribution
    @assert size(data) == size(likdists[1])
    simple = (gsamples === nothing) | (pri_dist === nothing) | (gdist === nothing)
    ℓW = simple ? map(p->_logweight(data, p), likdists) : map(zip(gsamples, likdists)) do (θg, ld)
        _logweight(data, θg, ld, pri_dist, gdist)
    end
    M = maximum(ℓW)
    W̃ = exp.(ℓW .- M)
    W̃ / sum(W̃) # normalize weights
end

"""
Compute Ê(θ∣y) using (normalized) importance weights W
"""
function importance_mean(W, gsamples)
    mapreduce((x, y)->x .* y, .+, gsamples, W)
end

function importance_mean(W, gsamples::Vector{T}) where T <: NamedTuple
    NamedTuple{keys(gsamples[1])}(importance_mean(W, values.(gsamples)))
end

"""
Effective sample size for (normalized) importance weights W
"""
importance_ess(W) = 1 / sum(x->x^2, W)

"""
Compute the model evidence p(y | d) using precomputed likelihood distributions or from calling `likelihood` on each `sim`
"""
function model_evidence(data, likdists::Vector{T}) where T <: Distribution
    map(d->pdf(d, data), likdists) |> mean
end

function model_evidence(data, sim::EnsembleSolution, likelihood::Function)
    map(x->pdf(likelihood(u), data), sim) |> mean
end

#= Some common likelihood constructors =#

joint_prior(pdist::AbstractDEParamDistribution) = product_distribution(vcat(values(random_vars(pdist))...))

function array_binom(n, p)
    @assert (length(n) == length(p)) || length(n) == 1 || length(p) == 1
    Binomial.(n, max.(1e-5, p))
end
joint_binom(n, p) = product_distribution(array_binom(n, p))

array_poisson(λs) = Poisson.(max.(1e-5, λs))
joint_poisson(λs) = product_distribution(array_poisson(λs))

array_poisson(n, p) = Poisson.(max.(1e-5, n .* p))
joint_poisson(n, p) = product_distribution(array_poisson(n, p))