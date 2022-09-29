
export marginal_divergence, δ

function _md_iter(y, xnum, xdenom, om::AbstractObservationModel)
    ℓnum = logpdf_particles(om, xnum, y)
    ℓdenom = logpdf_particles(om, xdenom, y)
    marginal_likelihood(ℓnum) - marginal_likelihood(ℓdenom)
end

function marginal_divergence(ϕ::Tuple, θfixed::NamedTuple, lm::LM, om::AbstractObservationModel; N=2000, saveat=1, dekwargs...) where {LM<:AbstractLatentModel}
    ϕtup = NamedTuple{ϕ}(map(x->get(θfixed, x, nothing), ϕ))
    xtrue = solve(lm, θfixed; saveat, dekwargs...).u
    ts = length(saveat) > 1 ? saveat : Int.(lm.start:saveat:lm.stop).+1
    y = Particles(N, observe_dist(om; observe_params(om, xtrue)...))[ts]
    xpart = solve(lm, ϕtup; saveat, dekwargs...).u
    xprior = solve(lm; saveat, dekwargs...).u
    return marginal_divergence(y, xpart, xprior, om)
end

δ(ϕ, θfixed, lm, om; N=2000, saveat=1, dekwargs...) = marginal_divergence(ϕ, θfixed, lm, om; N, saveat, dekwargs...)

function marginal_divergence(
    y::Vector{Particles{T, N}}, xpart::Union{Vector{<:AbstractFloat}, Vector{Particles{S, M}}}, 
    xprior::Vector{Particles{S, M}}, om::AbstractObservationModel
) where {T, S, N, M}
    md_iter = y->_md_iter(y, xpart, xprior, om)
    bymap(md_iter, y) |> pmean
end

# export all_designs_precomps, local_utility, local_marginal_utility

# nsse(θ, θest) = -sum((θ .- θest).^2)

# function initial_fit(x, pdist, likelihood; dekwargs...)
#     y₀ = rand(likelihood(x))
#     ch = sample_mcmc(y₀, pdist, likelihood; arraylik=false, dekwargs...)
#     fitchain(ch, pdist)
# end

# function get_nsse(y, θtrue, likdists; gsamples, pri_dist, gdist)
#     W = importance_weights(y, likdists; gsamples, pri_dist, gdist)
#     nsse(θtrue, importance_mean(W, gsamples))
# end

# function sig(y, pri_ldists, true_ldist)
#     log_evidence = marginal_likelihood(y, pri_ldists)
#     logpdf(true_ldist, y) - log_evidence
# end

# function sig(y, pri_ldists, true_ldists::Vector)
#     log_cond = marginal_likelihood(y, true_ldists) # marginalize out free param
#     log_evidence = marginal_likelihood(y, pri_ldists)
#     log_cond - log_evidence
# end

# function nsse_precomps(xtrue, pdist, likelihood=joint_poisson; Ng=2000, dekwargs...)  
#     ## fit a initial model for the IS sampling distribution
#     ## TODO this isn't a sound approach and should be fixed at some point
#     gdist = initial_fit(10 .* xtrue, pdist, likelihood; dekwargs...)
#     names = random_vars(pdist) |> keys
#     ## Get Ng samples from the target distribution g
#     postdraw = map(eachcol(rand(gdist, Ng))) do draw
#         NamedTuple{names}(draw)
#     end
#     gsims = simulate(postdraw, pdist; dekwargs...)
#     pri_dist = joint_prior(pdist)
#     gsamples = map(x->vcat(values(x)...), postdraw)
#     return (gsamples, pri_dist, gdist, gsims)
# end

# # function all_designs_precomps(
# #     θtrue, pdist::AbstractDEParamDistribution; # signal here is where lik get wrapped
# #     umap=(SIG=100, NSSE=100), dekwargs...
# # )
# #     ret = Dict()
# #     # xtrue = solve(de_problem(pdist, θtrue; dekwargs...), Tsit5()).u
# #     # if :NSSE ∈ keys(umap)
# #     #     nm = (:gsamples, :pri_dist, :gdist, :gsims)
# #     #     vals = nsse_precomps(xtrue, pdist, likelihood; Ng=umap.NSSE, dekwargs...)
# #     #     ret[:NSSE] = NamedTuple{nm}(vals)
# #     # end
# #     if :SIG ∈ keys(umap)
# #         psims = simulate(pdist, umap.SIG; dekwargs...)
# #         ret[:SIG] = (;psims)
# #     end
# #     return ret
# # end

# function local_utility(
#     θtrue, pdist::AbstractDEParamDistribution, dlik; # signal here is where lik get wrapped
#     N=100, precomps, dekwargs...
# )
#     xtrue = solve(de_problem(pdist, θtrue; dekwargs...), Tsit5()).u
#     ufunc = Dict()
#     if :NSSE ∈ keys(precomps)
#         gsamples, pri_dist, gdist, gsims = precomps[:NSSE]
#         ldist_nsse = map(dlik, gsims)
#         ufunc[:NSSE] = y->get_nsse(y, values(θtrue), ldist_nsse; gsamples, pri_dist, gdist)
#     end
#     if :SIG ∈ keys(precomps)
#         true_ldist = dlik(xtrue)
#         ldist_sig = map(dlik, precomps[:SIG].psims)
#         ufunc[:SIG] = y->sig(y, ldist_sig, true_ldist)
#     end

#     yreps = pmap(1:N) do _ # expectation over ys
#         y = rand(dlik(xtrue))
#         [f(y) for f ∈ values(ufunc)]
#     end
#     utils = [mean(getindex.(yreps, i)) for i=1:length(keys(ufunc))]
#     return (;zip(keys(ufunc), utils)...)
# end

# """
# Get the fixed obs params, i.e. the true ones that are not in `free_obs_params`
# """
# function fixed_obs_params(true_obs_params, free_obs_params)
#     props = collect(properties(true_obs_params))
#     f(tup) = !(tup[1] ∈ keys(free_obs_params[1]))
#     NamedTuple(filter(f, props))
# end

# """
# Optional precomputations
# free_obs: `NamedTuple`\n
# marginals: `Tuple`
# """
# function all_designs_precomps(
#     θtrue, pdist::PD; 
#     M=100, free_obs=nothing, marginals=nothing, dekwargs...
# ) where {PD<:AbstractDEParamDistribution}
#     ret = Dict{String, Any}(
#         "true_sim" => solve(de_problem(PD(;θtrue...); dekwargs...), Tsit5()).u,
#         "pri_sims" => simulate(pdist, M; dekwargs...)
#     )
#     # now the optional ones
#     if !isnothing(free_obs)
#         nm = keys(free_obs)
#         free_obs_params = [NamedTuple{nm}(rand.(values(free_obs))) for _ ∈ 1:M]
#         ret["free_obs_params"] = free_obs_params
#     end
#     if !isnothing(marginals) # hold specified param constant and simulate over others
#         for θᵢ ∈ marginals
#             props = properties(θtrue) |> collect
#             θnew = filter(tup->tup[1] == θᵢ, props) |> NamedTuple
#             margsim = simulate(PD(;θnew...), M; dekwargs...)
#             ret[string("margsim_", θᵢ)] = margsim
#         end
#     end
#     return ret
# end

# # function local_utility(
# #     true_sim, sims, true_obs_params, free_obs_params::Vector{T}, likelihood; N=100
# # ) where {T <: NamedTuple}
# #     true_ldist = likelihood(true_sim; true_obs_params...)
# #     obs_pfixed = fixed_obs_params(true_obs_params, free_obs_params)
# #     likdists = map(tup->likelihood(tup[1]; obs_pfixed..., tup[2]...), zip(sims, free_obs_params))

# #     ureps = pmap(1:N) do _ # expectation over ys
# #         y = rand(true_ldist)
# #         sig(y, likdists, true_ldist)
# #     end
# #     Û = mean(ureps)
# #     return Û
# # end

# function local_utility(
#     true_sim, pri_sims, likelihood::Function; 
#     N=100, M=100, obs_params=(;), distributed=nprocs()>1
# )

#     true_ldist = likelihood(true_sim; obs_params...)
#     bank_idxs = eachindex(pri_sims)

#     if distributed
#         ureps = pmap(1:N) do _ # expectation over ys
#             y = rand(true_ldist)
#             sig(y, pri_ldists, true_ldist)
#         end
#     else
#         pri_ldists = Vector(undef, N)
#         ureps = Vector{Float64}(undef, N)
#         Threads.@threads for i=1:N
#             pri_ldists[i] = map(s->likelihood(pri_sims[s]; obs_params...), sample(bank_idxs, M))
#             ureps[i] = sig(rand(true_ldist), pri_ldists[i], true_ldist)
#         end
#     end
#     return mean(ureps)
# end

# function local_marginal_utility(
#     true_sim, true_cond_sims, pri_sims, likelihood=(sim, m)->obs_t(sim, m, 1); 
#     N=100, M=100, true_obs_mod::OM=PoissonTests(100.), obs_mod::OM=true_obs_mod
# ) where OM <: ObservationModel

#     true_ldist = likelihood(true_sim, true_obs_mod)
#     bank_idxs = eachindex(true_cond_sims)
#     om_samps = [sample_obs_mod(obs_mod) for _=1:M]

#     marg_ldists = Vector(undef, N)
#     pri_ldists = Vector(undef, N)
#     ureps = Vector{Float64}(undef, N)
#     Threads.@threads for i=1:N
#         marg_ldists[i] = map((s, i)->likelihood(true_cond_sims[s], om_samps[i]), sample(bank_idxs, M), 1:M)
#         pri_ldists[i] = map((s, i)->likelihood(pri_sims[s], om_samps[i]), sample(bank_idxs, M), 1:M)
#         ureps[i] = sig(rand(true_ldist), pri_ldists[i], marg_ldists[i])
#     end
#     return mean(ureps)
# end

# function local_utility(precomps::Dict, likelihood; N=100) # Dict so u can bundle but still adjust e.g. t
#     local_utility(precomps.true_sim, precomps.sims, precomps.true_obs_params, precomps.free_obs_param, likelihood; N)
# end