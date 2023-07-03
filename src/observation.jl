
export AbstractObservationModel
export observe_params, observation, observe_dist, joint_observe_dist, logpdf_particles, log_likelihood, obs_info_mat

export PoissonRate, NConstVar

# TODO 6/20/23: it would be nice to have a "ragged replications" way to work similar to nreps but with diff. number of reps per design point

"""
Objects using the AbstractObservationModel interface should implement the functions
- `logpdf_particles` gives the log likelihood of `data` for each particle. If the observation model contains particles as well,
they are propogated jointly with μ. Note that you may just be able to use `logpdf` from `Distributions.jl` depending on your likelihood function,
although it will probably be slower
- `observe_dist` gives a vector of `Distribution`s corresponding to the likelihood of a true process μ

Implementing this interface then gives method extensions for multiple replications (adding `nreps` and supporting data as Matrix),
in addition to giving required ingredients for the RMD
"""
abstract type AbstractObservationModel{T} end

#= Method extensions =#
log_likelihood(m, μ, data) = logpdf_particles(m, μ, data)
log_likelihood(m::AbstractObservationModel, μ, data::Matrix) = sum(logpdf_particles(m, μ, data[:,i]) for i in 1:size(data, 2))

# function _observe_dist(m::AbstractObservationModel, μ::Vector{<:Particles})
#     @warn "Non-fixed μ. Using mean of partciles instead"
#     observe_dist(m, pmean.(μ))
# end

observe_dist(m, μ, nreps) = stack(fill(observe_dist(m, μ), nreps))

function joint_observe_dist(m::AbstractObservationModel, μ)
    product_distribution(observe_dist(m, μ))
end

joint_observe_dist(m, μ, nreps) = product_distribution(observe_dist(m, μ, nreps))

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

function logpdf_particles(m::PoissonRate, μ::Vector{<:Param{T}}, data::Vector) where {T}
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
    
observe_dist(m::NConstVar, μ) = Normal.(μ, sqrt(m.σ²))

function logpdf_particles(m::NConstVar, μ::Vector{<:Param{T}}, data::Vector) where T
    -length(data) / convert(T, 2) * (log(convert(T, 2π)) + log(m.σ²)) - (convert(T, 0.5)/m.σ²) * sum((μ .- data).^2)
end

# TODO: implement logistic regression model "Logistic"

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
