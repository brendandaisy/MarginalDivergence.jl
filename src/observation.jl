
export AbstractObservationModel
export vecindex
export observe_params, observation, observe_dist, joint_observe_dist, logpdf_particles, marginal_likelihood, obs_info_mat

export PoissonRate, NConstVar

"""
Objects using the AbstractObservationModel interface should implement the functions
- `logpdf_particles` gives the log likelihood of `data` for each particle. If the observation model contains particles as well,
they are propogated jointly with μ
- `observe_dist` gives a vector of `Distribution`s corresponding to the likelihood of a deterministic μ. If μ contains particles, 
`observe_dist` will be called with each mean with a warning

Implementing this interface then gives method extensions for multiple replications (adding `nreps` and supporting data as Matrix),
in addition to giving required ingredients for the RMD
"""
abstract type AbstractObservationModel{T} end

#= Method extensions =#
logpdf_particles(m::AbstractObservationModel, μ, data::Matrix) = sum(logpdf_particles(m, μ, data[:,i]) for i in 1:size(data, 2))

function observe_dist(m::AbstractObservationModel, μ::Vector{<:Particles})
    @warn "Non-fixed μ. Using mean of partciles instead"
    observe_dist(m, pmean.(μ))
end

observe_dist(m, μ, nreps) = stack(fill(observe_dist(m, μ), nreps))

function joint_observe_dist(m::AbstractObservationModel, μ)
    product_distribution(observe_dist(m, μ))
end

joint_observe_dist(m, μ, nreps) = fill(joint_observe_dist(m, μ), nreps)

#TODO move these?
"""
Compute the (log) marginal likelihood log(p(y)) using precomputed likelihood distributions
"""
function marginal_likelihood(log_lik::Particles{T, N}) where {T, N}
    m = convert(T, N)
    -log(m) + logsumexp(log_lik.particles)
end

# for the case when nothing is marginalized
marginal_likelihood(log_lik::AbstractFloat) = log_lik

#= Pre-provided Observation Models =#
  
"""
Poisson distributed observations, multiplying latent process by "testing" rate `η`

Testing is (possibly unknown) fixed rate across each observation time point
"""
struct PoissonRate{T} <: AbstractObservationModel{T}
    η::Param{T}
end

function observe_dist(m::PoissonRate, μ)
    Poisson.(m.η * μ)
end

function logpdf_particles(m::PoissonRate, μ::VecRealOrParticles{T, N}, data::Vector) where {T, N}
    f(λ, k) = -λ + k * log(λ) - convert(T, logfactorial(k))
    λs = m.η * μ
    sum(f(t[1], t[2]) for t in zip(λs, data))
end

obs_info_mat(m::PoissonRate, x) = Diagonal(m.η ./ x)

"""
Normal observations with constant variance
"""
struct NConstVar{T} <:AbstractObservationModel{T}
    σ²::Param{T}
end
    
observe_dist(m::NConstVar, μ) = Normal.(μ, m.σ²)

function logpdf_particles(m::NConstVar, μ::VecRealOrParticles{T, N}, data::Vector) where {T, N}
    -length(data) / 2 * (log(2π) + log(m.σ²)) - (0.5/m.σ²) * sum((μ .- data).^2)
end

# function ℓ_force_particles(p::AbstractArray{Particles{T, N}}, data, dfunc; joint=true) where {T, N}
#     if joint
#         l = x->logpdf(dfunc(x), data)
#     else
#         l = x->logpdf.(dfunc.(x), data)
#     end
#     ret = map(1:nparticles(p[1])) do i
#         x = vecindex.(p,i) # vector resulting from pushing single particle thru latent model
#         sum(l(x))
#     end
#     return Particles{T, N}(ret)
# end
