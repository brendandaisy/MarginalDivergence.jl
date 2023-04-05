export AbstractLatentModel, ODEModel, DDEModel
export peak_random_vars, allfixed, resample
export timespan, initial_values, parameters, hist_func, de_func, de_problem, solve

abstract type AbstractLatentModel{T} end
abstract type ODEModel{T} <: AbstractLatentModel{T} end
abstract type DDEModel{T} <: AbstractLatentModel{T} end

#= Fallback methods for AbstractLatentModel interface =#
timespan(::AbstractLatentModel) = error("timespan function not implemented")
initial_values(::AbstractLatentModel) = error("initial_values function not implemented")
parameters(::AbstractLatentModel) = error("parameters function not implemented")
de_func(::AbstractLatentModel) = error("de_func function not implemented")
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
Conveniently produce an `ODEProblem` from `m` by automatically matching `initial_values`, `timespan` which can be solved, etc.

Using `solve(::AbstractLatentModel, ...)` directly is preferred
"""
function de_problem(m::ODEModel; dekwargs...)
    init = initial_values(m)
    ts = timespan(m)
    p = parameters(m)
    ODEProblem(de_func(m), init, ts, p; dekwargs...)
end

# function de_problem(m::DDEModel; dekwargs...)
#     init = initial_values(m)
#     ts = timespan(m)
#     p = parameters(m)
#     h = hist_func(m)
#     DDEProblem(de_func(m), init, h, ts, p; dekwargs...)
# end

# TODO: honestly, this shouldn't be the go to solve since it is making a new object. Not that big of a deal since have particles though
function CommonSolve.solve(m::M; alg=Tsit5(), dekwargs...) where M <: AbstractLatentModel
    solve(de_problem(m; dekwargs...), alg)
end

"""
Return a solution to the latent model, where parameters in `θnew` replace the existing parameters in the model.

This is useful, e.g. for quickly solving a version of the model where all params have been fixed.
"""
function CommonSolve.solve(m::M, θnew::NamedTuple; alg=Tsit5(), dekwargs...) where M <: AbstractLatentModel
    # θnew = convert_tuple(m.start, θnew)
    props = properties(m) |> collect
    θrest = filter(tup->tup[1] ∉ keys(θnew), props) |> NamedTuple
    mnew = M(;θnew..., θrest...)
    solve(de_problem(mnew; dekwargs...), alg)
end