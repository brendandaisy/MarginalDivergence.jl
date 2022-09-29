module DiffEqInformationTheory

using CommonSolve, OrdinaryDiffEq, DelayDiffEq
using MonteCarloMeasurements
using Distributions
using Parameters
using SpecialFunctions
using IterTools
# import Statistics: mean
using StatsFuns

export Param

Param{T} = Union{T, Particles{T}} where T <: Real

(::Type{Param{T}})(x::Real) where T <: Real = T(x)

#= Misc =#

convert_tuple(::T, t) where T <: Real = (;zip(keys(t), convert.(T, values(t)))...)

#= Include package components =#

include("latent.jl")
include("observation.jl")
include("inference.jl")
include("information.jl")

include("epi-models.jl")

end # module
