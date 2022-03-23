

abstract type ObservationModel end

struct PoissonTests <: ObservationModel
    ntest::Float64
end

likelihood(sim, mod::PoissonTests) = product_distribution(Poisson.(mod.ntest .* vec(sim)))

struct PoissonBias <: ObservationModel
    ntest::Float64
    b::Distribution
    popsize::Float64
end

function likelihood(sim, mod::PoissonBias; b=rand(mod.b))
    λ₊ = sim * mod.ntest * (mod.popsize/mod.ntest)^b
    Poisson.(λ₊)
end

function single_obs(sim, mod::PoissonBias, t; b=rand(mod.b))
    λ₊ = sim[t] * mod.ntest * (mod.popsize/mod.ntest)^b
    Poisson(λ₊)
end

function inct_obs(sim, mod::PoissonBias, tspan; b=rand(mod.b))
    λ₊ = sim[1:tspan] * mod.ntest * (mod.popsize/mod.ntest)^b
    Poisson.(λ₊)
end
    

