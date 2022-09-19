module DEParamDistributions

using OrdinaryDiffEq, DelayDiffEq
using MonteCarloMeasurements
using Distributions
using Parameters
import DiffEqBase: solve
import IterTools: fieldvalues, properties
# using Distributed
import Statistics: mean
# using Turing #TODO: perhaps we could make it a point of avoiding using Turing? (and make pull requests to existing julia pkg for this)
# using Lazy
import StatsFuns: logsumexp

export Param

export AbstractLatentModel, ODEModel, DDEModel
export timespan, initial_values, parameters, hist_func, de_func, de_problem, solve
# export match_initial_values, match_parameters, random_vars, param_sample, de_problem, solve_de_problem
# export sample_de_problem, sample_de_problem!, update_de_problem!, remake_prob

Param{T} = Union{T, Particles{T}} where T <: Real

(::Type{Param{T}})(x::Real) where T <: Real = T(x)

# struct Parameter{T}
#     val::Union{T, Vector{T}}
# end

# isfixed(p::Param) = !(p isa Particles)
# random_vars(v::Vector{Param}) = filter(isfixed, v)

abstract type AbstractLatentModel{T} end
abstract type ODEModel{T} <: AbstractLatentModel{T} end
abstract type DDEModel{T} <: AbstractLatentModel{T} end

#= Convenience wrappers for AbstractDEParamDistribution interface =#

timespan(::AbstractLatentModel) = error("timespan function not implemented")
initial_values(::AbstractLatentModel) = error("initial_values function not implemented")
parameters(::AbstractLatentModel) = error("parameters function not implemented")
de_func(::AbstractLatentModel) = error("de_func function not implemented")
hist_func(::DDEModel) = error("hist_func function not implemented")

#TODO: The NamedTuple obsession could just be from nice printing and "peak" methods; keep internals fast as possible

# """
# Return a `NamedTuple` of properties in pdist which are a `Particles`
# """
# function peak_random_vars(pdist::AbstractDEParamDistribution)
#     props = collect(properties(pdist))
#     NamedTuple(filter(x->isa(x[2], Particles), props))
# end

# allfixed(pdist::AbstractDEParamDistribution) = length(random_vars(pdist)) == 0

function de_problem(m::ODEModel; dekwargs...)
    init = initial_values(m)
    ts = timespan(m)
    p = parameters(m)
    ODEProblem(de_func(m), init, ts, p; dekwargs...)
end

# function de_problem(pdist::DDEParamDistribution, params=(;); dekwargs...)
#     init = match_initial_values(pdist, params)
#     ts = timespan(pdist)
#     p = match_parameters(pdist, params)
#     h = hist_func(pdist)
#     DDEProblem(de_func(pdist), init, h, ts, p; dekwargs...)
# end

# TODO: solve is not needed! Instead, 
function solve(m::AbstractLatentModel; alg=Tsit5(), dekwargs...)
    solve(de_problem(m; dekwargs...), alg)
end

# function sample_de_problem(pdist::AbstractDEParamDistribution; dekwargs...)
#     ps = param_sample(pdist)
#     de_problem(pdist, ps; dekwargs...)
# end

# ## TODO: no check for incompat. prob and pdist
# function sample_de_problem!(prob::SciMLBase.SciMLProblem, pdist::AbstractDEParamDistribution)
#     # if isa(pdist.start, Distribution) | isa(pdist.stop, Distribution)
#     #     error("must make new ODEProblem for timespan distributions")
#     # end
#     ps = param_sample(pdist)
#     update_de_problem!(prob, pdist, ps)
# end

# function update_de_problem!(prob::SciMLBase.SciMLProblem, pdist, params)
#     prob.u0 .= match_initial_values(pdist, params)
#     prob.p .= match_parameters(pdist, params)
#     prob
# end

# function remake_prob(prob::SciMLBase.SciMLProblem, pdist, params)
#     remake(prob; u0=match_initial_values(pdist, params), p=match_parameters(pdist, params))
# end

#= Include other methods for simulation and inference =#

include("epi-models.jl")
# include("simulation.jl")
# include("observation.jl")
# include("inference.jl")
# include("utility.jl")

end # module
