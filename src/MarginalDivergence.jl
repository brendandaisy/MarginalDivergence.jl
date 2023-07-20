module MarginalDivergence

using CommonSolve, OrdinaryDiffEq
using MonteCarloMeasurements
using Distributions
using Parameters
using SpecialFunctions
using IterTools
# import Statistics: mean
using StatsFuns
using ForwardDiff
using LinearAlgebra
using KissMCMC

export Param, particles_index

"""
The `Param` type is a variable in dynamical system model. It may be a fixed or random variable of numerical precision `T <: Real`
"""
Param{T} = Union{T, Particles{T}} where T <: Real

(::Type{Param{T}})(x::Real) where T <: Real = T(x) # functor to convert objects of type `T <: Real` to `Param{T}`

#= Misc methods =#
"""
Get the `i`th particle from each particle distribution in an array

Multiple indexing is not supported
"""
particles_index(p, i::Int) = MonteCarloMeasurements.vecindex(p, i)

"""
Convert all values of a tuple `t` to a given type. Returns a (named) tuple.
"""
convert_tuple(::T, t) where T <: Real = (;zip(keys(t), convert.(T, values(t)))...)

#= Include package components =#

include("latent.jl")
include("observation.jl")
include("information.jl")

include("epi-models.jl")

end # module
