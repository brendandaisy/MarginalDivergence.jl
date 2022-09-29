## Methods for posterior approximation θ∣y

# export mle
# export importance_weights, importance_weights!, importance_mean, importance_ess
export marginal_likelihood

# function mle_optim(data, pdist::AbstractDEParamDistribution, likelihood=Poisson; names=keys(random_vars(pdist)), dekwargs...)
#     prob = de_problem(typeof(pdist); dekwargs...)
#     f = θ -> log_likelihood(data, prob, θ, likelihood, names)
#     init = (getfield(pdist, k) for k ∈ names)
#     init_float = [p isa Distribution ? rand(p) : p for p ∈ init]
#     optimize(f, init_float, BFGS())
# end


# #= Methods for importance sampling =#

# # _logweight(y, u, jointlik::Function) = logpdf(jointlik(u), y)
# _logweight(y, dist) = @inbounds logpdf(dist, y)

# # function _logweight(y, g, gx; lf, pd, gd=pd)
# #     logpdf(lf(gx), y) + logpdf(pd, g) - logpdf(gd, g)
# # end

# function _logweight(y, g, ld, pd, gd)
#     @inbounds logpdf(ld, y) + logpdf(pd, g) - logpdf(gd, g)
# end

# """
# Compute numerically stable, normalized importance weights given a sampling distribution 𝐺(𝜃)

# If any keywords missing, assume 𝐺(𝜃) = 𝑃(𝜃) and use the simplified formula
# """
# function importance_weights(
#     data, likdists::Vector{T}; 
#     gsamples=nothing, pri_dist=nothing, gdist=nothing
# ) where T <: Distribution
#     @assert size(data) == size(likdists[1])
#     simple = (gsamples === nothing) | (pri_dist === nothing) | (gdist === nothing)
#     ℓW = simple ? map(p->_logweight(data, p), likdists) : map(zip(gsamples, likdists)) do (θg, ld)
#         _logweight(data, θg, ld, pri_dist, gdist)
#     end
#     M = maximum(ℓW)
#     W̃ = exp.(ℓW .- M)
#     W̃ / sum(W̃) # normalize weights
# end

# """
# Compute Ê(θ∣y) using (normalized) importance weights W
# """
# function importance_mean(W, gsamples)
#     mapreduce((x, y)->x .* y, .+, gsamples, W)
# end

# function importance_mean(W, gsamples::Vector{T}) where T <: NamedTuple
#     NamedTuple{keys(gsamples[1])}(importance_mean(W, values.(gsamples)))
# end

# """
# Effective sample size for (normalized) importance weights W
# """
# importance_ess(W) = 1 / sum(x->x^2, W)

"""
Compute the (log) marginal likelihood log(p(y | d)) using precomputed likelihood distributions or from calling `likelihood` on each `sim`
"""
function marginal_likelihood(log_lik::Particles{T, N}) where {T, N}
    m = convert(T, N)
    -log(m) + logsumexp(log_lik.particles)
end

# for the case when nothing is marginalized (TODO not very good naming convention...)
marginal_likelihood(log_lik::AbstractFloat) = log_lik

# function marginal_likelihood(data, sim::EnsembleSolution, likelihood::Function)
#     map(x->pdf(likelihood(u), data), sim) |> mean
# end