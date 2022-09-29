
export AbstractObservationModel
export vecindex
export observe_params, observe_dist, logpdf_particles

export PoissonTests, ℓ_ind_poisson

vecindex = MonteCarloMeasurements.vecindex

#= Generating particles from an observation process =#

# TODO for now, assume that no uncertainty is created from the observation model, but this can be extended by modifying the latent process 
# pushing things through ODE + modification with @bymap, as you tested


abstract type AbstractObservationModel{T} end

observe_params(::AbstractObservationModel, x::Vector{<:Real}) = (μ=x,)
observe_params(m::AbstractObservationModel, x, ts) = observe_params(m, x[ts])
# observe_params(x, m::AbstractObservationModel, t) = observe(m, x[t])

observe_dist(::AbstractObservationModel; μ::Vector{S}) where {S<:AbstractFloat} = MvNormal(μ, 0.1)
# observe_dist(m::AbstractObservationModel, ts; params...) where {S<:AbstractFloat} = observe_dist(m; μ[ts])

struct PoissonTests{T} <: AbstractObservationModel{T}
    ntest::Param{T}
end

observe_params(m::PoissonTests, x::Vector{<:Param{T}}) where T<:Real = (λ=map(xt->xt * m.ntest, x),)
logpdf_particles(m::PoissonTests, x::Vector{<:Param{T}}, data) where T<:Real= ℓ_ind_poisson(observe_params(m, x).λ, data, T)
observe_dist(::PoissonTests; λ) = product_distribution(Poisson.(Float64.(λ))) # convert because rand won't work otherwise

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
