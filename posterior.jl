using Turing

## TODO decide how to track multiple compartments and make consistent with prior-predict
@model function odemodel(
        y, prior::T1, likelihood, 
        saveat=0:(length(y)-1), ::Type{T2}=Float64;
        var_transform=identity,
        vars=random_vars(prior)
    ) where {T1 <: AbstractODEParamDistribution, T2 <: Real}

    theta = Vector{T2}(undef, length(vars))
    for (i, val) ∈ enumerate(vars)
        theta[i] ~ NamedDist(val, keys(vars)[i]) # name of distr goes last
    end

    samp_vals = var_transform(convert.(T2, theta))
    prob = ode_problem(prior, NamedTuple(zip(keys(vars), samp_vals)))
    sol = solve(prob, Tsit5(), save_idxs=1, saveat=saveat)
    if sol.retcode != :Success
        Turing.@addlogprob!(-Inf)
        return
    end
    y ~ arraydist(likelihood(sol.u))
    return sol.u
end

function joint_binom(n, p)
    @assert (length(n) == length(p)) || length(n) == 1 || length(p) == 1
    Binomial.(n, max.(1e-5, p))
end

# prob = ODEProblem(sir!, [0.01, 0.1], (0., 30.), [0.3, 0.1, 1.])
# sim = solve(prob, Tsit5(); save_idxs=1)
# inf = sim(1:30).u
# y = [rand(Binomial(100, x)) for x in inf]
# plot(inf)
# scatter!(y / 100)

# tpri = [TruncatedNormal(.1, .01, 0, 1), TruncatedNormal(.3, .1, 0, .4)]

# sir_pdist = SIRParamDistribution(30., TruncatedNormal(.1, .01, 0, 1), TruncatedNormal(.3, .1, 0, .4), 0.1)
# tmod = sir_model(y, sir_pdist);
# chain = sample(tmod, NUTS(.9), MCMCThreads(), 1000, 1)
# inf_samp = generated_quantities(tmod, chain);
