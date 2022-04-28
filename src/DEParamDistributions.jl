module DEParamDistributions

using OrdinaryDiffEq, DelayDiffEq
using Distributions
import IterTools: fieldvalues, properties
using Distributed
import Statistics: mean
using Turing
using StatsBase
import StatsFuns: logsumexp

export TParam, AbstractDEParamDistribution, ODEParamDistribution, DDEParamDistribution
export timespan, initial_values, parameters, hist_func, de_func
export match_initial_values, match_parameters, random_vars, param_sample, de_problem, solve_de_problem
export sample_de_problem, sample_de_problem!, update_de_problem!, remake_prob

const TParam = Union{Float64, Distribution}

abstract type AbstractDEParamDistribution end
abstract type ODEParamDistribution <: AbstractDEParamDistribution end
abstract type DDEParamDistribution <: AbstractDEParamDistribution end

#= Convenience wrappers for AbstractDEParamDistribution interface =#

timespan(pdist::AbstractDEParamDistribution) = (pdist.start, pdist.stop)
initial_values(::AbstractDEParamDistribution) = error("initial_values method not implemented")
parameters(::AbstractDEParamDistribution) = error("parameters method not implemented")
de_func(::AbstractDEParamDistribution) = error("de_func method not implemented")
hist_func(::DDEParamDistribution) = error("hist_func method not implemented")

#= Matching values and making ODEProblem instances =#

### TODO support random timespans with .* (0, 0) trick

"""
Return a `NamedTuple` of the initial_values from pdist, or from params if they appear there.

The returned values of type `eltype(params)`
"""
function match_initial_values(pdist::AbstractDEParamDistribution, params)
    length(params) == 0 ? ret = Float64[] : ret = Vector{eltype(params)}()
    for (k, v) ∈ properties(pdist)
        if k ∉ initial_values(pdist)
            continue
        end
        k ∈ keys(params) ? push!(ret, params[k]) : push!(ret, v)
    end
    ret
end

"""
Return a `NamedTuple` of the parameters from pdist, or from params if they appear there.

The returned values of type `eltype(params)`
"""
function match_parameters(pdist::AbstractDEParamDistribution, params)
    length(params) == 0 ? ret = Float64[] : ret = Vector{eltype(params)}()
    for (k, v) ∈ properties(pdist)
        if k ∉ parameters(pdist)
            continue
        end
        k ∈ keys(params) ? push!(ret, params[k]) : push!(ret, v)
    end
    ret
end

"""
Return a `NamedTuple` of properties in pdist which are a `Distribution`
"""
function random_vars(pdist::T) where T <: AbstractDEParamDistribution
    props = collect(properties(pdist))
    NamedTuple(filter(x->isa(x[2], Distribution), props))
end

isfixed(pdist::AbstractDEParamDistribution) = length(random_vars(pdist)) == 0

"""
Return a `NamedTuple` of a sample from `random_vars`
"""
function param_sample(pdist::AbstractDEParamDistribution)
    rvs = random_vars(pdist)
    NamedTuple{keys(rvs)}(rand.(values(rvs)))
end

"""
Initialize a differential equations problem of appropriate type (ODE, DDE).

Provide a type `T` to initialize a problem using `T`'s default fields.
"""
function de_problem(::Type{T}; dekwargs...) where {T <: AbstractDEParamDistribution}
    de_problem(T(); dekwargs...)
end

function de_problem(pdist::ODEParamDistribution, params=(;); dekwargs...)
    init = match_initial_values(pdist, params)
    ts = timespan(pdist)
    p = match_parameters(pdist, params)
    ODEProblem(de_func(pdist), init, ts, p; dekwargs...)
end

function de_problem(pdist::DDEParamDistribution, params=(;); dekwargs...)
    init = match_initial_values(pdist, params)
    ts = timespan(pdist)
    p = match_parameters(pdist, params)
    h = hist_func(pdist)
    DDEProblem(de_func(pdist), init, h, ts, p; dekwargs...)
end

function solve_de_problem(pdist::AbstractDEParamDistribution, params; alg=Tsit5(), dekwargs...)
    solve(de_problem(pdist, params; dekwargs...), alg)
end

function sample_de_problem(pdist::AbstractDEParamDistribution; dekwargs...)
    ps = param_sample(pdist)
    de_problem(pdist, ps; dekwargs...)
end

## TODO: no check for incompat. prob and pdist
function sample_de_problem!(prob::SciMLBase.SciMLProblem, pdist::AbstractDEParamDistribution)
    # if isa(pdist.start, Distribution) | isa(pdist.stop, Distribution)
    #     error("must make new ODEProblem for timespan distributions")
    # end
    ps = param_sample(pdist)
    update_de_problem!(prob, pdist, ps)
end

function update_de_problem!(prob::SciMLBase.SciMLProblem, pdist, params)
    prob.u0 .= match_initial_values(pdist, params)
    prob.p .= match_parameters(pdist, params)
    prob
end

function remake_prob(prob::SciMLBase.SciMLProblem, pdist, params)
    remake(prob; u0=match_initial_values(pdist, params), p=match_parameters(pdist, params))
end

#= Include other methods for simulation and inference =#

include("epi-odes.jl")
include("simulation.jl")
include("observation.jl")
include("inference.jl")
include("utility.jl")

end # module
