## Methods for posterior approximation θ∣y

export prior_predict
export turingode, sample_mcmc, mcmc_mean, fitchain, extract_vars
export mle
export importance_weights, importance_weights!, importance_mean, importance_ess
export marginal_likelihood
export joint_prior, array_poisson, joint_poisson, array_binom, joint_binom, array_neg_binom, joint_neg_binom

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
    sampler=NUTS(), iter=500, chains=4, arraylik=true, ŷ=false, dekwargs...
)
    prob₀ = sample_de_problem(pdist; dekwargs...)
    model = turingode(data, prob₀, pdist, likelihood; arraylik)
    ch = sample(model, sampler, MCMCThreads(), iter, chains)
    if ŷ
        return ch, generated_quantities(model, ch)
    end
    return ch
end

mcmc_mean(fit) = summarystats(fit).nt.mean

"""
Return a vector of `NamedTuple` of posterior draws from `ch`.

TODO this is broken!!!
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

###

bound_check(θ) = all(θ .≥ 0)

## current assumption is that model likelihood is indep in y and t
## data and likelihood should have matching types
## data vec should be vec(data), where data has matching dimensions of sol
function log_likelihood(data_vec, prob, θ, likelihood, names)
    if !bound_check(θ)
        return 1_000_000
    end
    update_de_problem!(prob, pdist, NamedTuple(zip(names, samp_vals)))
    sol = solve(prob, Tsit5())
    if sol.retcode != :Success
        return 1_000_000
    end
    lik = product_distribution(likelihood.(vec(Array(sol)))) # lol
    return -logpdf(lik, data_vec)
end

function mle_optim(data, pdist::AbstractDEParamDistribution, likelihood=Poisson; names=keys(random_vars(pdist)), dekwargs...)
    prob = de_problem(typeof(pdist); dekwargs...)
    f = θ -> log_likelihood(data, prob, θ, likelihood, names)
    init = (getfield(pdist, k) for k ∈ names)
    init_float = [p isa Distribution ? rand(p) : p for p ∈ init]
    optimize(f, init_float, BFGS())
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
Compute the (log) marginal likelihood log(p(y | d)) using precomputed likelihood distributions or from calling `likelihood` on each `sim`
"""
function marginal_likelihood(data, likdists::Vector{T}) where T <: Distribution
    -log(length(likdists)) + logsumexp(map(dist->logpdf(dist, data), likdists))
end

# function marginal_likelihood(data, sim::EnsembleSolution, likelihood::Function)
#     map(x->pdf(likelihood(u), data), sim) |> mean
# end

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

array_neg_binom(r::T, μ) where T <: Real = NegativeBinomial.(r, r ./ (r .+ μ))
joint_neg_binom(r, μ) = product_distribution(array_neg_binom(r, μ))