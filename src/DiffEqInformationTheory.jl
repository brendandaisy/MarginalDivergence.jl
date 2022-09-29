module DiffEqInformationTheory

using OrdinaryDiffEq, DelayDiffEq
using MonteCarloMeasurements
using Distributions
using Parameters
using SpecialFunctions
import IterTools: fieldvalues, properties
import Statistics: mean
import StatsFuns: logsumexp

export Param

Param{T} = Union{T, Particles{T}} where T <: Real

(::Type{Param{T}})(x::Real) where T <: Real = T(x)

#= Misc =#

convert_tuple(::T, t) where T <: Real = (;zip(keys(t), convert.(T, values(t)))...)

#= Include package components =#

include("epi-models.jl")
include("latent.jl")
include("observation.jl")
include("inference.jl")
include("information.jl")

end # module
