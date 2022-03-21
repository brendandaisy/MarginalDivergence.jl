
export simulate

# pluck_vecvec(vecvec, keep=1) = [x[keep] for x ∈ vecvec]

# function mean_timeseries(sim, ts; keep=1)
#     et = ensemble_timeseries(sim, ts; keep=keep)
#     getindex.(et)
# end

# function ensemble_timeseries(sim, ts; keep=1)
#     ret = []
#     for (θ, sol) ∈ sim
#         push!(ret, (param_list=θ, series=pluck_vecvec(sol(ts).u, keep)))
#     end
#     ret
# end

# function sample_curve(series, y_dist, y_dist_args)
#     [rand(y_dist(s, arg)) for (s, arg) ∈ zip(series, y_dist_args)]
# end

# function ensemble_timeseries(sim, ts, y_dist, y_dist_args; keep=1)
#     ret = []
#     for (θ, sol) ∈ sim
#         cc = pluck_vecvec(sol(ts).u, keep)
#         push!(ret, (param_list=θ, series=cc, y=sample_curve(cc, y_dist, y_dist_args)))
#     end
#     ret
# end

# function ensemble_timeseries(sim, t_inc::Float64, y_dist, y_dist_args; keep=1)
#     start, stop = sim[1][2].prob.tspan
#     ts = start:t_inc:stop
#     ret = []
#     for (θ, sol) ∈ sim
#         cc = pluck_vecvec(sol(ts).u, keep)
#         push!(ret, (param_list=θ, series=cc, y=sample_curve(cc, y_dist, y_dist_args)))
#     end
#     ret
# end

function simulate(
    pdist::AbstractDEParamDistribution, N=4000; keep=false, dekwargs...
)
    init_prob = sample_de_problem(pdist; dekwargs...)
    pf = (prob, i, repeat) -> sample_de_problem!(prob, pdist)
    if keep
        of = (prob, i) -> (prob, false) 
    else
        of = (prob, i) -> (prob.u, false)
    end
    solve(EnsembleProblem(init_prob, prob_func=pf, output_func=of), Tsit5(), EnsembleThreads(), trajectories=N)
end

## Run using precomputed vector of parameters
## pdist is just for matching names
function simulate(
    psamples::AbstractVector, pdist::AbstractDEParamDistribution; keep=false, dekwargs...
)
    init_prob = sample_de_problem(pdist; dekwargs...)
    pf = (prob, i, repeat) -> update_de_problem!(prob, pdist, psamples[i])
    if keep
        of = (prob, i) -> (prob, false) 
    else
        of = (prob, i) -> (prob.u, false)
    end
    solve(EnsembleProblem(init_prob, prob_func=pf, output_func=of), Tsit5(), EnsembleThreads(), trajectories=length(psamples))
end

function simulate(θfix::NamedTuple, pdist::PD, N; keep=false, dekwargs...) where {PD<:AbstractDEParamDistribution}
    props = properties(random_vars(pdist)) |> collect
    θrest = filter(tup->tup[1] ∉ keys(θfix), props) |> NamedTuple
    simulate(PD(;θrest..., θfix...), N; keep, dekwargs...)
end

## above func seems quite useful
## notice that different param_comb and marginal computations overlap! But in kinda complicated way
## plan: 1. maintain a dict of diff combs of free/fixed sims
## 2. this can be saved, and "runorloaded" during an actual run (actually could do this for full precomp...)
## 3. make use of these appropriately. For example, if param_comb=(β, α), compute p(y∣S₀) with marg for S₀ and p()