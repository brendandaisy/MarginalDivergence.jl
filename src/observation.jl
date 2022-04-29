
export ObservationModel, PoissonTests, PoissonBiasMult
export sample_obs_mod
export obs_t, obs_tspan

abstract type ObservationModel end

function sample_obs_mod(mod::OM) where OM <: ObservationModel
    v = collect(fieldvalues(mod))
    newv = map(x->isa(x, Distribution) ? rand(x) : x, v)
    OM(newv...)
end

function param_sample(mod::ObservationModel)
    v = collect(fieldvalues(mod))
    map(x->isa(x, Distribution) ? rand(x) : x, v)
end

struct PoissonTests <: ObservationModel
    ntest::TParam
end

function obs_t(sim, mod::PoissonTests, t)
    # η = param_sample(mod)
    Poisson(mod.ntest * sim[t])
end
function obs_tspan(sim, mod::PoissonTests, t)
    # η = param_sample(mod)
    joint_poisson(mod.ntest .* sim[1:t])
end

struct PoissonBiasMult <: ObservationModel
    ntest::TParam
    b::TParam
end

function obs_t(sim, mod::PoissonBiasMult, t)
    # η, b = param_sample(mod)
    Poisson(mod.b * mod.ntest * sim[t])
end
function obs_tspan(sim, mod::PoissonBiasMult, t)
    joint_poisson(mod.b * mod.ntest .* sim[1:t])
end

###

# struct PoissonTests <: ObservationModel
#     ntest::Float64
# end

# likelihood(sim, mod::PoissonTests) = product_distribution(Poisson.(mod.ntest .* vec(sim)))

# struct PoissonBias <: ObservationModel
#     ntest::Float64
#     b::Distribution
#     popsize::Float64
# end

# function likelihood(sim, mod::PoissonBias; b=rand(mod.b))
#     λ₊ = sim * mod.ntest * (mod.popsize/mod.ntest)^b
#     Poisson.(λ₊)
# end

# function single_obs(sim, mod::PoissonBias, t; b=rand(mod.b))
#     λ₊ = sim[t] * mod.ntest * (mod.popsize/mod.ntest)^b
#     Poisson(λ₊)
# end

# function inct_obs(sim, mod::PoissonBias, tspan; b=rand(mod.b))
#     λ₊ = sim[1:tspan] * mod.ntest * (mod.popsize/mod.ntest)^b
#     Poisson.(λ₊)
# end
    

