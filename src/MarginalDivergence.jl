module MarginalDivergence

using CommonSolve, OrdinaryDiffEq
using MonteCarloMeasurements
using Distributions
using Parameters
using SpecialFunctions
using IterTools
# import Statistics: mean
using StatsFuns

export Param, VecRealOrParticles

"""
The `Param` type is a variable in dynamical system model. It may be a fixed or random variable of numerical precision `T <: Real`
"""
Param{T} = Union{T, Particles{T}} where T <: Real

(::Type{Param{T}})(x::Real) where T <: Real = T(x) # functor to convert objects of type `T <: Real` to `Param{T}`

VecRealOrParticles{T, N} = Union{Vector{T}, Vector{Particles{T, N}}} where {T <: Real, N}

#= Misc methods =#

"""
Convert all values of a tuple `t` to a given type. Returns a (named) tuple.

Useful for enforcing 
"""
convert_tuple(::T, t) where T <: Real = (;zip(keys(t), convert.(T, values(t)))...)

#= Include package components =#

include("latent.jl")
include("observation.jl")
include("information.jl")

include("epi-models.jl")

end # module
