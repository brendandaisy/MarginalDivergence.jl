
export marginal_divergence, δ, mean_marginal_likelihood

function _md_iter(y, xnum, xdenom, om::AbstractObservationModel)
    ℓnum = logpdf_particles(om, xnum, y)
    ℓdenom = logpdf_particles(om, xdenom, y)
    marginal_likelihood(ℓnum) - marginal_likelihood(ℓdenom)
end

"""
The Restricted Marginal Divergence (RMD). If ϕ are The RMD is the difference between marginal log likelihoods of these two,
on average over data `y` originating from a true process.
"""
function marginal_divergence(ϕ::Tuple, θfixed::NamedTuple, lm::LM, om::AbstractObservationModel; N=3000, saveat=1, dekwargs...) where {LM<:AbstractLatentModel}
    ϕtup = NamedTuple{ϕ}(map(x->get(θfixed, x, nothing), ϕ))
    xtrue = solve(lm, θfixed; saveat, dekwargs...).u
    ts = length(saveat) > 1 ? saveat : Int.(lm.start:saveat:lm.stop) .+ 1
    y = Particles(N, observe_dist(om; observe_params(om, xtrue)...))[ts]
    xcond = solve(lm, ϕtup; saveat, dekwargs...).u
    xprior = solve(lm; saveat, dekwargs...).u
    return marginal_divergence(y, xcond, xprior, om)
end

"""
The Restricted Marginal Divergence (RMD). `xcond` is simulations sampled from the latent process, with a quantity of interest held fixed,
and `xprior` is simulations sampled from the full prior latent process. The RMD is the difference between marginal log likelihoods of these two,
on average over data `y` originating from a true process.
"""
function marginal_divergence(
    y::VecOrMat, xcond::VecRealOrParticles{T, S}, xprior::VecRealOrParticles{T, S}, om::AbstractObservationModel; N=3000
) where {T, S}
    mds = zeros(N)
    Threads.@threads for i=1:N
        mds[i] = _md_iter(rand.(y), xcond, xprior, om)
    end
    mean(mds)
end

"""
Compute expected log likelihood, marginalizing over whatever simulations are represented by `x`, with the expectation over `y`

Could be useful e.g. if you want to compute the RMD of several variables, while reusing the calculation for the full marginal likelihood E[P(y)]
"""
function mean_marginal_likelihood(
    y::Vector{Particles{T, N}}, x::VecRealOrParticles{S, M}, om::AbstractObservationModel
) where {T, S, N, M}
    ml_iter = y->marginal_likelihood(logpdf_particles(om, x, y))
    bypmap(ml_iter, y) |> pmean
end