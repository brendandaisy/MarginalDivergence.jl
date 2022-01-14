export sir!, SIRParamDistribution

function sir!(dx, x, p, t)
    I, R = x
    β, α, N = p
    dx[1] = (β / N) * I * (N - I - R) - α * I
    dx[2] = α * I
end

function seir!(dx, x, p)
    E, I, R = x
    β, α, γ, μ, N = p
    dx[1] = β * I * (N - E - I - R) - (γ + μ) * E
    dx[2] = γ * E - α * I
    dx[3] = μ * E + α * I
end

struct SIRParamDistribution <: AbstractODEParamDistribution
    start::Union{Float64, Distribution}
    stop::Union{Float64, Distribution}
    inf_init::Union{Float64, Distribution}
    rec_init::Union{Float64, Distribution}
    inf_rate::Union{Float64, Distribution}
    rec_rate::Union{Float64, Distribution}
    pop_size::Union{Float64, Distribution}
end

SIRParamDistribution(stop, rec_init, inf_rate, rec_rate) = 
    SIRParamDistribution(0.0, stop, 0.01, rec_init, inf_rate, rec_rate, 1.0)

initial_values(::Type{SIRParamDistribution}) = (:inf_init, :rec_init)
parameters(::Type{SIRParamDistribution}) = (:inf_rate, :rec_rate, :pop_size)
ode_func(::Type{SIRParamDistribution}) = sir!