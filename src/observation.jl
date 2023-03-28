
export AbstractObservationModel
export vecindex
export observe_params, observation, observe_dist, joint_observe_dist, logpdf_particles, marginal_likelihood

export PoissonRate, ℓ_ind_poisson, NConstVar

vecindex = MonteCarloMeasurements.vecindex

## TODO I think this is still a work in progress; some tweaks look halfway done 3/20

#= Generating particles from an observation process =#

abstract type AbstractObservationModel{T} end

# observation(::AbstractObservationModel, x) = error("`observe params` not implemented")
# logpdf_particles(::AbstractObservationModel, x, data) = error("`logpdf_particles` not implemented")
logpdf_particles(m::AbstractObservationModel, x::VecRealOrParticles, data::Matrix) = sum(logpdf_particles(m, x, data[:,i]) for i in 1:size(data, 2))

observe_params(m::AbstractObservationModel, x, ts) = observe_params(m, x[ts])

observe_dist(::AbstractObservationModel, x::Vector{<:Particles}) = error("currently only scalar x supported")

# TODO: particle x - what is the correct way? marginalize (how to correctly?)
function observe_dist(m::AbstractObservationModel, x::Vector{<:Real})
    obs_p = observe_params(m, x)
    map(t->observation(m, t...), values(obs_p)...)
end

observe_dist(m, x, nreps) = stack(fill(observe_dist(m, x), nreps))

function joint_observe_dist(m::AbstractObservationModel, x::VecRealOrParticles)
    product_distribution(observe_dist(m, x))
end

joint_observe_dist(m, x, nreps) = fill(joint_observe_dist(m, x), nreps)

# _marg_obs_param(x) = Float64.(x) # default no marginalization (when x is fixed vector)
# _marg_obs_param(x::Vector{<:Particles}) = Float64.(pmean.(x)) # convert because rand may not work otherwise

"""
Compute the (log) marginal likelihood log(p(y | d)) using precomputed likelihood distributions or from calling `likelihood` on each `sim`
"""
function marginal_likelihood(log_lik::Particles{T, N}) where {T, N}
    m = convert(T, N)
    -log(m) + logsumexp(log_lik.particles)
end

# for the case when nothing is marginalized (TODO not very good naming convention...)
marginal_likelihood(log_lik::AbstractFloat) = log_lik

#= Pre-provided Observation Models =#
  
"""
Poisson distributed observations, multiplying latent process by "testing" rate `η`

Testing is (possibly unknown) fixed rate across each observation time point
"""
struct PoissonRate{T} <: AbstractObservationModel{T}
    η::Param{T}
end

observation(::PoissonRate, λ) = Poisson(λ)
observe_params(m::PoissonRate, x::VecRealOrParticles) = (λ=x * m.η,)
logpdf_particles(m::PoissonRate, x::VecRealOrParticles{T, N}, data::Vector) where {T, N} = ℓ_ind_poisson(observe_params(m, x).λ, data, T)

function ℓ_ind_poisson(λs, ks, ::Type{T}=Float64) where T
    f(λ, k) = -λ + k * log(λ) - convert(T, logfactorial(k))
    sum(f(t[1], t[2]) for t in zip(λs, ks))
end

"""
Normal observations with constant variance
"""
struct NConstVar{T} <:AbstractObservationModel{T}
    σ²::Param{T}
end
    
observation(m::NConstVar, μ) = Normal(μ, m.σ²)
observe_params(m::NConstVar, x::VecRealOrParticles) = (μ=x,)

function logpdf_particles(m::NConstVar, x::VecRealOrParticles{T, N}, data::Vector) where {T, N}
    μvec = observe_params(m, x)
    -length(y) / 2 * (log(2π) + log(m.σ²)) - (0.5/m.σ²) * sum((μvec .- data).^2)
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
