#= MCMC approximation using Turing =#

# TODO I don't really have plans to integrate this with Particles implementation, but just in case

@model function turingode(
    data, prob, pdist::T1, likelihood=array_poisson, ::Type{T2}=Float64;
    var_transform=identity, vars=random_vars(pdist), arraylik=true
) where {T1 <: AbstractDEParamDistribution, T2 <: Real}

theta = Vector{T2}(undef, length(vars))
for (i, val) ∈ enumerate(vars)
    theta[i] ~ NamedDist(val, keys(vars)[i]) # name of distr goes last
end

samp_vals = var_transform(convert.(T2, theta))
pr = remake_prob(prob, pdist, NamedTuple(zip(keys(vars), samp_vals)))
sol = solve(pr, Tsit5())
if sol.retcode != :Success
    Turing.@addlogprob!(-Inf)
    return
end
arraylik ? data ~ arraydist(likelihood(sol.u)) : data ~ likelihood(sol.u)
return sol.u
end

"""
Create a Turing model and perform sampling using algorithm `sampler`.

Sampling uses `MCMCThreads` and returns a `Chains` object.
"""
function sample_mcmc(
data, pdist::AbstractDEParamDistribution, likelihood=array_poisson; 
sampler=NUTS(), iter=500, chains=4, arraylik=true, ŷ=false, dekwargs...
)
prob₀ = sample_de_problem(pdist; dekwargs...)
model = turingode(data, prob₀, pdist, likelihood; arraylik)
ch = sample(model, sampler, MCMCThreads(), iter, chains)
if ŷ
    return ch, generated_quantities(model, ch)
end
return ch
end

mcmc_mean(fit) = summarystats(fit).nt.mean

"""
Return a vector of `NamedTuple` of posterior draws from `ch`.

TODO this is broken!!!
"""
function extract_vars(ch::Chains, pdist::AbstractDEParamDistribution)
vars = keys(random_vars(pdist))
tup = Turing.get(ch, [v for v in vars]) # get named tuple of axisarrays
map(eachindex(tup[1])) do i
    NamedTuple{vars}([t[i] for t in tup])
end
end

"""
Approximate posterior draws from `ch` using `Distributions.fit`.

Dimensions of the resulting distribution are ordered to "match" `pdist`.
"""
function fitchain(ch::Chains, pdist::AbstractDEParamDistribution; d=MvNormal)
rvname = keys(random_vars(pdist))
ii = indexin(ch.name_map.parameters, vcat(rvname...)) ## indexer to make cols match with pdist
Distributions.fit(d, Array(ch)[:,ii]')
end