export AbstractLatentModel, ODEModel, DDEModel
export peak_random_vars, allfixed
export timespan, initial_values, parameters, hist_func, de_func, de_problem, solve

abstract type AbstractLatentModel{T} end
abstract type ODEModel{T} <: AbstractLatentModel{T} end
abstract type DDEModel{T} <: AbstractLatentModel{T} end

#= Convenience wrappers for AbstractDEParamDistribution interface =#

timespan(::AbstractLatentModel) = error("timespan function not implemented")
initial_values(::AbstractLatentModel) = error("initial_values function not implemented")
parameters(::AbstractLatentModel) = error("parameters function not implemented")
de_func(::AbstractLatentModel) = error("de_func function not implemented")
hist_func(::DDEModel) = error("hist_func function not implemented")

"""
Return a `NamedTuple` of properties in pdist which are a `Particles`
"""
function peak_random_vars(lm::AbstractLatentModel)
    props = collect(properties(lm))
    NamedTuple(filter(x->isa(x[2], Particles), props))
end

allfixed(lm::AbstractLatentModel) = !has_particles(lm)

function de_problem(m::ODEModel; dekwargs...)
    init = initial_values(m)
    ts = timespan(m)
    p = parameters(m)
    ODEProblem(de_func(m), init, ts, p; dekwargs...)
end

function de_problem(m::DDEModel; dekwargs...)
    init = initial_values(m)
    ts = timespan(m)
    p = parameters(m)
    h = hist_func(m)
    DDEProblem(de_func(m), init, h, ts, p; dekwargs...)
end

function solve(m::AbstractLatentModel; alg=Tsit5(), dekwargs...)
    DiffEqBase.solve(de_problem(m; dekwargs...), alg)
end

function solve(m::M, θnew::NamedTuple; alg=Tsit5(), dekwargs...) where M <: AbstractLatentModel
    θnew = convert_tuple(m.start, θnew)
    props = properties(m) |> collect
    θrest = filter(tup->tup[1] ∉ keys(θnew), props) |> NamedTuple
    mnew = M(;θnew..., θrest...)
    DiffEqBase.solve(de_problem(mnew; dekwargs...), alg)
end