export SIRParamDistribution, DSEIRParamDistribution

function sir!(dx, x, p, t)
    S, I = x
    β, α = p
    dx[1] = -β * I * S
    dx[2] = β * I * S - α * I
end

# function seir!(dx, x, p)
#     E, I, R = x
#     β, α, γ, μ, N = p
#     dx[1] = β * I * (N - E - I - R) - (γ + μ) * E
#     dx[2] = γ * E - α * I
#     dx[3] = μ * E + α * I
# end

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

Base.@kwdef struct DSEIRParamDistribution <: DDEParamDistribution
    start::Float64 = 0.
    stop::Float64 = 30.
    S₀::TParam = 0.99
    E₀::TParam = 0.
    I₀::TParam = 0.01
    β::TParam = 0.3
    α::TParam = 0.1
    τ::TParam = 1.
end

# TODO Edel should be -h(;idxs=1), i.e. the newly exp ind at t - τ
function delay_seir!(du, u, h, p, t)
    β, α, τ = p
    S, E, I = u
    Edel = h(p, t - τ; idxs=2)
    du[1] = -β * S * I
    du[2] = max(β * S * I - Edel, -E)
    du[3] = Edel - α * I
end

hdseir(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0. : [0.99, 0., 0.01]

initial_values(::DSEIRParamDistribution) = (:S₀, :E₀, :I₀)
parameters(::DSEIRParamDistribution) = (:β, :α, :τ)
de_func(::DSEIRParamDistribution) = delay_seir!
hist_func(::DSEIRParamDistribution) = hdseir