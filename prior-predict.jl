using DifferentialEquations.EnsembleAnalysis

export prior_predict, predict_yhat
## TODO: would be less confusing if it was the same syntax for pripd/postpd to:
## get vector of series
## get summary stats of series

## TODO: make dealing with >1 compartment consistent with posterior

## Apply lik to each vector (timeseries/sols) of a vector (samples)
function predict_yhat(vec, jointlik)
    curves = vec
    if vec[1] isa ODESolution
        curves = [sol.u for sol ∈ vec]
    end
    [rand.(dcurve) for dcurve ∈ jointlik.(curves)] # draw from each of the joint lik curves
end

pluck_vecvec(vecvec, keep=1) = [x[keep] for x ∈ vecvec]

function mean_timeseries(sim, ts; keep=1)
    et = ensemble_timeseries(sim, ts; keep=keep)
    getindex.(et)
end

function ensemble_timeseries(sim, ts; keep=1)
    ret = []
    for (θ, sol) ∈ sim
        push!(ret, (param_list=θ, series=pluck_vecvec(sol(ts).u, keep)))
    end
    ret
end

function sample_curve(series, y_dist, y_dist_args)
    [rand(y_dist(s, arg)) for (s, arg) ∈ zip(series, y_dist_args)]
end

function ensemble_timeseries(sim, ts, y_dist, y_dist_args; keep=1)
    ret = []
    for (θ, sol) ∈ sim
        cc = pluck_vecvec(sol(ts).u, keep)
        push!(ret, (param_list=θ, series=cc, y=sample_curve(cc, y_dist, y_dist_args)))
    end
    ret
end

function ensemble_timeseries(sim, t_inc::Float64, y_dist, y_dist_args; keep=1)
    start, stop = sim[1][2].prob.tspan
    ts = start:t_inc:stop
    ret = []
    for (θ, sol) ∈ sim
        cc = pluck_vecvec(sol(ts).u, keep)
        push!(ret, (param_list=θ, series=cc, y=sample_curve(cc, y_dist, y_dist_args)))
    end
    ret
end

## TODO: sparse saving for speed, at sacrifice of summaries and plotting
function prior_predict(pdist::T, N=4000; save_idxs=1, saveat=1., de_kwargs...) where T <: AbstractODEParamDistribution
    init_prob = sample_ode_problem(pdist; save_idxs=save_idxs, saveat=saveat, de_kwargs...)
    pf = (prob, i, repeat) -> sample_ode_problem!(prob, pdist)
    solve(EnsembleProblem(init_prob, prob_func=pf), Tsit5(), EnsembleThreads(), trajectories=N)
end