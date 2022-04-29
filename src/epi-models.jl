export SIRParamDistribution, SEIRParamDistribution, DSEIRParamDistribution

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
Base.@kwdef struct SIRParamDistribution <: ODEParamDistribution
    start::Float64 = 0.
    stop::Float64 = 30.
    S₀::TParam = 0.99
    I₀::TParam = 0.01
    β::TParam = 0.3
    α::TParam = 0.1
end

initial_values(::SIRParamDistribution) = (:S₀, :I₀)
parameters(::SIRParamDistribution) = (:β, :α)
de_func(::SIRParamDistribution) = sir!

#= SEIR Model =#

function seir!(dx, x, p)
    S, E, I = x
    β, α, γ, μ = p
    dx[1] = -β * I * S
    dx[2] = β * I * S - (γ + μ) * E
    dx[3] = γ * E - α * I
end

"""
A standard implementation of the SEIR model, with incubation period 1/γ and chance for not becoming infectious μ
"""
Base.@kwdef struct SEIRParamDistribution <: ODEParamDistribution
    start::Float64 = 0.
    stop::Float64 = 30.
    S₀::TParam = 0.99
    E₀::TParam = 0.01
    I₀::TParam = 0.
    β::TParam = 0.3
    γ::TParam = 0.3
    μ::TParam = 0.
    α::TParam = 0.1
end

initial_values(::SEIRParamDistribution) = (:S₀, :E₀, :I₀)
parameters(::SEIRParamDistribution) = (:β, :γ, :μ, :α)
de_func(::SEIRParamDistribution) = seir!

#= Delay SEIR =#

"""
An SEIR model using delay diff equations. Exposures from a incubation delay t - τ then become infectious
"""
Base.@kwdef struct DSEIRParamDistribution <: DDEParamDistribution
    start::Float64 = 0.
    stop::Float64 = 30.
    S₀::TParam = 0.99
    E₀::TParam = 0.01
    I₀::TParam = 0.
    β::TParam = 0.3
    α::TParam = 0.1
    τ::TParam = 1.
end

function delay_seir!(du, u, h, p, t)
    β, α, τ = p
    S, E, I = u
    Edel = -h(p, t - τ; idxs=1)
    du[1] = -β * S * I
    du[2] = max(β * S * I - Edel, -E)
    du[3] = Edel - α * I
end

hdseir(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0. : [0.99, 0.01, 0.]

initial_values(::DSEIRParamDistribution) = (:S₀, :E₀, :I₀)
parameters(::DSEIRParamDistribution) = (:β, :α, :τ)
de_func(::DSEIRParamDistribution) = delay_seir!
hist_func(::DSEIRParamDistribution) = hdseir