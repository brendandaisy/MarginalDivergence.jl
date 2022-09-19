export SIRModel, SEIRModel, DSEIRModel

#= SIR Model =#

function sir!(dx, x, p, t)
    S, I = x
    β, α = p
    dx[1] = -β * I * S
    dx[2] = β * I * S - α * I
end

"""
SIR model with infection intensity β and S+I+R=1
"""
@with_kw struct SIRModel{T<:Real} <: ODEModel{T} @deftype Param{T}
    start::T = 0.
    stop::T = 30.
    S₀ = 0.99
    I₀ = 0.01
    β = 0.3
    α = 0.1
    @assert start <= stop
    # @assert 0. <= S₀ + I₀ <= 1.
    # @assert all([β, α] .>= 0.)
end

# Override the untyped default constructor
# SIRModel(;start::T=0., stop::T=30., S₀::T=0.99, I₀::T=0.01, β::T=0.3, α::T=0.1) where T <: Real = SIRModel(start, stop, S₀, I₀, β, α)
# SIRModel
# SIRModel(t, T, s, i, b, a) = SIRModel{Float64}(t, T, s, i, b, a)

timespan(m::SIRModel) = (m.start, m.stop)
initial_values(m::SIRModel) = [m.S₀, m.I₀]
parameters(m::SIRModel) = [m.β, m.α]
de_func(::SIRModel) = sir!

#= SEIR Model =#

# function seir!(dx, x, p, t)
#     S, E, I = x
#     β, α, γ, μ = p
#     dx[1] = -β * I * S
#     dx[2] = β * I * S - (γ + μ) * E
#     dx[3] = γ * E - α * I
# end

# """
# A standard implementation of the SEIR model, with incubation period 1/γ and chance for not becoming infectious μ
# """
# Base.@kwdef struct SEIRParamDistribution{T} <: ODEParamDistribution{T}
#     start::Param{T} = T(0.)
#     stop::Param{T} = T(30.)
#     S₀::Param{T} = T(0.99)
#     E₀::Param{T} = T(0.01)
#     I₀::Param{T} = T(0.)
#     β::Param{T} = T(0.3)
#     γ::Param{T} = T(0.3)
#     μ::Param{T} = T(0.)
#     α::Param{T} = T(0.1)
# end

# initial_values(::SEIRParamDistribution) = (:S₀, :E₀, :I₀)
# parameters(::SEIRParamDistribution) = (:β, :γ, :μ, :α)
# de_func(::SEIRParamDistribution) = seir!

# #= Delay SEIR =#

"""
An SEIR model using delay diff equations. Exposures from a incubation delay t - τ then become infectious
"""
# Base.@kwdef struct DSEIRParamDistribution{T} <: DDEParamDistribution{T}
#     start::Param{T} = T(0.)
#     stop::Param{T} = T(30.)
#     S₀::Param{T} = T(0.99)
#     E₀::Param{T} = T(0.01)
#     I₀::Param{T} = T(0.)
#     β::Param{T} = T(0.3)
#     α::Param{T} = T(0.1)
#     τ::Param{T} = T(1.)
# end

# function delay_seir!(du, u, h, p, t)
#     β, α, τ = p
#     S, E, I = u
#     Edel = -h(p, t - τ; idxs=1)
#     du[1] = -β * S * I
#     du[2] = max(β * S * I - Edel, -E)
#     du[3] = Edel - α * I
# end

# hdseir(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0. : [0.99, 0.01, 0.]

# initial_values(::DSEIRParamDistribution) = (:S₀, :E₀, :I₀)
# parameters(::DSEIRParamDistribution) = (:β, :α, :τ)
# de_func(::DSEIRParamDistribution) = delay_seir!
# hist_func(::DSEIRParamDistribution) = hdseir