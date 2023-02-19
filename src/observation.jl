
export AbstractObservationModel
export vecindex
export observe_params, observation, observe_dist, logpdf_particles, marginal_likelihood

export PoissonRate, ℓ_ind_poisson

vecindex = MonteCarloMeasurements.vecindex

#= Generating particles from an observation process =#

abstract type AbstractObservationModel{T} end

observe_params(::AbstractObservationModel, x) = error("`observe params` not implemented")
observation(::AbstractObservationModel, x) = error("`observation` not implemented")
logpdf_particles(::AbstractObservationModel, x, data) = error("`logpdf_particles` not implemented")

observe_params(m::AbstractObservationModel, x, ts) = observe_params(m, x[ts])

function observe_dist(m::AbstractObservationModel, x::VecRealOrParticles)
    obs_p = observe_params(m, x)
    obs_p_avg = map(_marg_obs_param, values(obs_p))
    product_distribution(observation(m, obs_p_avg...))
end

_marg_obs_param(x) = Float64.(x) # default no marginalization (when x is fixed vector)
_marg_obs_param(x::Vector{<:Particles}) = Float64.(pmean.(x)) # convert because rand may not work otherwise

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
Poisson distributed observations, multiplying latent process by "testing" rate `ntest`

Testing is (possibly unknown) fixed rate across each observation time point
"""
struct PoissonRate{T} <: AbstractObservationModel{T}
    ntest::Param{T}
end

observation(::PoissonRate, λ) = Poisson.(λ)
observe_params(m::PoissonRate, x::VecRealOrParticles) = (λ=x * m.ntest,)
logpdf_particles(m::PoissonRate, x::VecRealOrParticles{T, N}, data) where {T, N} = ℓ_ind_poisson(observe_params(m, x).λ, data, T)

function ℓ_ind_poisson(λs, ks, ::Type{T}=Float64) where T
    f(λ, k) = -λ + k * log(λ) - convert(T, logfactorial(k))
    sum(t->f(t[1], t[2]), zip(λs, ks))
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
