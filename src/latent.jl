export AbstractLatentModel, GenericModel, DiffEqModel, ODEModel, DDEModel
export peak_random_vars, allfixed, resample
export timespan, initial_values, parameters, hist_func, de_func, de_problem, solve

abstract type AbstractLatentModel{T} end

#=
Interface 1: Generic models
requires implementing a `solve` function which is entirely responsible with creating
sensible output from the struct's values
=#
abstract type GenericModel{T} <: AbstractLatentModel{T} end

#=
Interface 2: Differential equation models
requires implementing at least `timespan` `initial_values` `parameters` and `de_func` which point 
to the relavent componenets of DifferentialEquations.jl's implementation. 
The `solve` method as well as some other helpers are then inherited
=#
abstract type DiffEqModel{T} <: AbstractLatentModel{T} end
abstract type ODEModel{T} <: DiffEqModel{T} end
abstract type DDEModel{T} <: DiffEqModel{T} end

#= Fallback methods for DiffEqModel interface =#
timespan(::DiffEqModel) = error("timespan function not implemented")
initial_values(::DiffEqModel) = error("initial_values function not implemented")
parameters(::DiffEqModel) = error("parameters function not implemented")
de_func(::DiffEqModel) = error("de_func function not implemented")
hist_func(::DDEModel) = error("hist_func function not implemented")

"""
Return a `NamedTuple` of properties in `lm` which are a `Particles`
"""
function peak_random_vars(lm::AbstractLatentModel)
    props = collect(properties(lm))
    NamedTuple(filter(x->isa(x[2], Particles), props))
end

function Base.string(lm::LM) where {LM<:AbstractLatentModel}
    valstr = join(fieldvalues(lm), ", ")
    "$LM($valstr)" 
end

"""
Whether `lm` does not contain any fields that are a `Particles`, recursively
"""
allfixed(lm::AbstractLatentModel) = !has_particles(lm)

# MonteCarloMeasurements.nparticles(lm::AbstractLatentModel) = allfixed(lm) ? 0 : nparticles(peak_random_vars(lm)[1])

function resample(lm::LM, n) where LM <: AbstractLatentModel
    vars = peak_random_vars(lm)
    newvars = NamedTuple{keys(vars)}(bootstrap(vcat(vars...), n))
    LM(;newvars...)
end

"""
Produce an `ODEProblem` from `m` by automatically matching `initial_values`, `timespan` which can be solved, etc.

Using `solve(::AbstractLatentModel, ...)` directly is preferred
"""
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

function CommonSolve.solve(m::LM, θnew::NamedTuple; kwargs...) where LM <: GenericModel
    props = properties(m) |> collect
    θrest = filter(tup -> tup[1] ∉ keys(θnew), props) |> NamedTuple
    mnew = LM(;θnew..., θrest...)
    solve(mnew; kwargs...)
end

function CommonSolve.solve(m::LM; alg=Tsit5(), dekwargs...) where LM <: DiffEqModel
    solve(de_problem(m; dekwargs...), alg)
end

"""
Return a solution to the latent model, where parameters in `θnew` replace the existing parameters in the model.

This is useful, e.g. for quickly solving a version of the model where all params have been fixed.
"""
function CommonSolve.solve(m::LM, θnew::NamedTuple; alg=Tsit5(), dekwargs...) where LM <: DiffEqModel
    # θnew = convert_tuple(m.start, θnew)
    props = properties(m) |> collect
    θrest = filter(tup->tup[1] ∉ keys(θnew), props) |> NamedTuple
    mnew = LM(;θnew..., θrest...)
    solve(de_problem(mnew; dekwargs...), alg)
end