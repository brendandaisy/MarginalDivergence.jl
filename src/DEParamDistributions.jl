module DEParamDistributions

export AbstractODEParamDistribution
export timespan, match_initial_values, match_parameters, random_vars, ode_problem, sample_ode_problem, sample_ode_problem!, update_ode_problem!
export prior_predict
export initial_values, parameters, SIRParamDistribution

using OrdinaryDiffEq
using Distributions
import IterTools: fieldvalues, properties
import NamedTupleTools: ntfromstruct
# using ParameterizedFunctions

abstract type AbstractODEParamDistribution end

## Methods to implement AbstractODEParamDistribution interface

timespan(pdist::AbstractODEParamDistribution) = (pdist.start, pdist.stop)
initial_values(pdist::AbstractODEParamDistribution) = initial_values(typeof(pdist))
parameters(pdist::AbstractODEParamDistribution) = parameters(typeof(pdist))
ode_func(pdist::AbstractODEParamDistribution) = ode_func(typeof(pdist))

## Matching values and making ODEProblem instances

# Return named tuple initial values from pdist, or from params if they appear there
function match_initial_values(pdist::AbstractODEParamDistribution, params)
    ret = Vector{eltype(params)}()
    for (k, v) ∈ properties(pdist)
        if !(k ∈ initial_values(pdist))
            continue
        end
        k ∈ keys(params) ? push!(ret, params[k]) : push!(ret, v)
    end
    ret
end

function match_parameters(pdist::AbstractODEParamDistribution, params)
    ret = Vector{eltype(params)}()
    for (k, v) ∈ properties(pdist)
        if !(k ∈ parameters(pdist))
            continue
        end
        k ∈ keys(params) ? push!(ret, params[k]) : push!(ret, v)
    end
    ret
end

function random_vars(pdist::T) where T <: AbstractODEParamDistribution
    props = collect(properties(pdist))
    NamedTuple(filter(x->isa(x[2], Distribution), props))
end

function param_sample(pdist::AbstractODEParamDistribution)  
    kk = fieldnames(typeof(pdist))
    vv = map(fieldvalues(pdist)) do v
        isa(v, Distribution) ? rand(v) : v
    end
    NamedTuple(zip(kk, vv))
end

function ode_problem(::Type{T}, tspan=(0., 1.); kwargs...) where {T <: AbstractODEParamDistribution}
    init = Vector{Float64}(undef, length(initial_values(T)))
    ts = tspan
    p = Vector{Float64}(undef, length(parameters(T)))
    ODEProblem(ode_func(T), init, ts, p; kwargs...)
end

function ode_problem(pdist, params; kwargs...)
    init = match_initial_values(pdist, params)
    ts = timespan(pdist)
    p = match_parameters(pdist, params)
    ODEProblem(ode_func(typeof(pdist)), init, ts, p; kwargs...)
end

# TODO remove or add check for only fixed params in pdist
ode_problem(pdist; kwargs...) = ode_problem(pdist, ntfromstruct(pdist); kwargs...)

function sample_ode_problem(pdist::AbstractODEParamDistribution; kwargs...)
    ps = param_sample(pdist)
    ode_problem(pdist, ps; kwargs...)
end

## TODO: no check for incompat. prob and pdist
function sample_ode_problem!(prob::ODEProblem, pdist::AbstractODEParamDistribution)
    if isa(pdist.start, Distribution) | isa(pdist.stop, Distribution)
        error("must make new ODEProblem for timespan distributions")
    end
    ps = param_sample(pdist)
    update_ode_problem!(prob, pdist, ps)
end

function update_ode_problem!(prob::ODEProblem, pdist, params)
    prob.u0 .= match_initial_values(pdist, params)
    prob.p .= match_parameters(pdist, params)
    prob
end

## Include other methods for simulation and inference

include("epi-odes.jl")
include("prior-predict.jl")
include("posterior.jl")

end # module
