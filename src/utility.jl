using DEParamDistributions
using OrdinaryDiffEq
using Distributions
using Distributed
import Statistics: mean

export all_designs_precomps, local_utility

nsse(θ, θest) = -sum((θ .- θest).^2)

sig(true_ldist, y, evidence) = logpdf(true_ldist, y) - log(evidence)

function initial_fit(x, pdist, likelihood; dekwargs...)
    y₀ = rand(likelihood(x))
    ch = sample_mcmc(y₀, pdist, likelihood; arraylik=false, dekwargs...)
    fitchain(ch, pdist)
end

function get_nsse(y, θtrue, likdists; gsamples, pri_dist, gdist)
    W = importance_weights(y, likdists; gsamples, pri_dist, gdist)
    nsse(θtrue, importance_mean(W, gsamples))
end

function get_sig(y, likdists, true_ldist)
    evidence = model_evidence(y, likdists)
    sig(true_ldist, y, evidence)
end

function nsse_precomps(xtrue, pdist, likelihood=joint_poisson; Ng=2000, dekwargs...)  
    ## fit a initial model for the IS sampling distribution
    ## TODO this isn't a sound approach and should be fixed at some point
    gdist = initial_fit(10 .* xtrue, pdist, likelihood; dekwargs...)
    names = random_vars(pdist) |> keys
    ## Get Ng samples from the target distribution g
    postdraw = map(eachcol(rand(gdist, Ng))) do draw
        NamedTuple{names}(draw)
    end
    gsims = simulate(postdraw, pdist; dekwargs...)
    pri_dist = joint_prior(pdist)
    gsamples = map(x->vcat(values(x)...), postdraw)
    return (gsamples, pri_dist, gdist, gsims)
end

function all_designs_precomps(
    θtrue, pdist::AbstractDEParamDistribution, likelihood=joint_poisson; # signal here is where lik get wrapped
    umap=(SIG=100, NSSE=100), dekwargs...
)
    ret = Dict()
    xtrue = solve(de_problem(pdist, θtrue; dekwargs...), Tsit5()).u
    if :NSSE ∈ keys(umap)
        nm = (:gsamples, :pri_dist, :gdist, :gsims)
        vals = nsse_precomps(xtrue, pdist, likelihood; Ng=umap.NSSE, dekwargs...)
        ret[:NSSE] = NamedTuple{nm}(vals)
    end
    if :SIG ∈ keys(umap)
        psims = simulate(pdist, umap.SIG; dekwargs...)
        ret[:SIG] = (;psims)
    end
    return ret
end

function local_utility(
    d, θtrue, pdist::AbstractDEParamDistribution, dlik=x->joint_poisson(d, x); # signal here is where lik get wrapped
    N=100, precomps, dekwargs...
)
    xtrue = solve(de_problem(pdist, θtrue; dekwargs...), Tsit5()).u
    
    # if :NSSE ∈ keys(precomps)
        gsamples, pri_dist, gdist, gsims = precomps[:NSSE]
        ldist_nsse = map(dlik, gsims)
        _nsse = y->get_nsse(y, values(θtrue), ldist_nsse; gsamples, pri_dist, gdist)
    # end
    # if :SIG ∈ keys(precomps)
        true_ldist = dlik(xtrue)
        ldist_sig = map(dlik, precomps[:SIG].psims)
        _sig = y->get_sig(y, ldist_sig, true_ldist)
    # end

    utils = pmap(1:N) do _ # expectation over ys
        y = rand(dlik(xtrue))
        [_sig(y), _nsse(y)]
    end
    return (SIG=mean(first.(utils)), NSSE=mean(last.(utils)))
end