using Revise
using Test
using DEParamDistributions
using MonteCarloMeasurements, Distributions

function joint_cond(α, β, R)
    pdf(Uniform(0.1, 0.4), α) * pdf(Uniform(0.3, 2.), β) * α/β * pdf(Uniform(0.1, 0.9), α*R/β)
end

function accept_reject(sampler, target; M=100, N=100)
    ret = []
    for _ in 1:N
        s = rand(sampler)
        u = rand()
        while u >= target(s) / (M*pdf(sampler, s))
            s = rand(sampler)
            u = rand()
        end
        push!(ret, s)
    end
    return (first.(ret), last.(ret))
end

α = Uniform(0.1, 0.4)
β = Uniform(0.3, 2.)

αcond, βcond = accept_reject(product_distribution([α, β]), x->joint_cond(x[1], x[2], 3.))
Scond = (αcond * 3) ./ βcond

(βcond .* Scond) ./ αcond

density(βcond)

a = collect(0.1:0.001:0.4)
pa = α_cond.(a, 0.9)

# data information gain,
# predictive information information gain

αsamp = sample(a, Weights(pa), 100_000; replace=true)
βsamp = .9*αsamp

b = collect(0.3:0.001:2.0)
pb = β_cond.(b, 0.9)

βsamp2 = sample(b, Weights(pb), 100_000; replace=true)
αsamp2 = βsamp2/0.9

histogram(αsamp; bins=15, normalize=true)
histogram(αsamp2; bins=15, normalize=true)
histogram2d(αsamp, βsamp)

var(αsamp)
var(αsamp2)

m_cond_r = SIRModel(β=Particles(βsamp2), α=Particles(αsamp))
sol = DEParamDistribtions.solve(m_cond_r; save_idxs=2, saveat=1)

@testset "Biased Coins" begin
    struct Coins <: AbstractObservationModel{Int8} end
    DEParamDistributions.logpdf_particles(::Coins, x::Vector{<:Particles}, data) = sum(t->t[2]*log(t[1]) + (1 - t[2])*log(1-t[1]), zip(x, data))
    DEParamDistributions.observe_dist(::Coins, x::Vector{<:AbstractFloat}) = product_distribution(Bernoulli.(x))

    om = Coins()
    x = outer_product(fill(Uniform(0, 1), 2), 10_000)
    @test observe_params(om, x) === x

    ℓlik = logpdf_particles(om, x, [0, 0])
    @test marginal_likelihood(ℓlik) ≈ log(1/4)

    xtrue = [0, 0.5]
    ytrue = Particles(observe_dist(om, xtrue))
    xpart = [0, Particles(10_000, Uniform(0, 1))]

    md = marginal_divergence(ytrue, xpart, x, om)
    # ytrue is [0, 0] or [0,1], both of which have MℓL=log(1/2), so 
    @test md ≈ (log(1/2) - log(1/4))
end

@testset "Sanity checks with SIR model and marginal divergence" begin
    m = SIRModel(S₀=0..1, β=Particles(TruncatedNormal(0.3, 0.1, 0, Inf)))
    inf = solve(m; save_idxs=2, saveat=1).u
    om = PoissonTests(100)

    λsamp = observe_params(om, inf, 1:2)
    ysamp = rand(observe_dist(om; λ=100*Matrix(inf)[1,1:2]), 100_000)
    μy = mean.(eachrow(ysamp))
    @test isapprox(μy, Matrix(λsamp.λ)[1,:]; atol=0.05)

    md1 = marginal_divergence((:β, :S₀), (S₀=0.7, β=0.3), m, om; save_idxs=2, saveat=1)
    md2 = marginal_divergence((:S₀,), (S₀=0.7, β=0.3), m, om; save_idxs=2, saveat=1)

    mtrue = SIRModel(S₀=0.7, β=0.3)
    inf_true = solve(mtrue; save_idxs=2, saveat=1).u
    ytrue = Particles(observe_dist(om; λ=observe_params(om, inf_true).λ))

    mpart = SIRModel(S₀=0.7, β=Particles(TruncatedNormal(0.3, 0.1, 0, Inf)))
    infpart = solve(mpart; save_idxs=2, saveat=1).u

    md3 = marginal_divergence(ytrue, infpart, inf, om)

    @test isapprox(md2, md3; atol=0.05)
    @test md1 > md2
end

@testset "PoissonTest with unknown testing rate" begin
    m = SIRModel(S₀=0..1, β=Particles(TruncatedNormal(0.3, 0.1, 0, Inf)))
    inf = solve(m; save_idxs=2, saveat=1).u
    om = PoissonTests(Particles(DiscreteUniform(100, 200)))

    p = observe_params(om, inf)
    ysamp = rand(observe_dist(om; λ=Matrix(p.λ)[1,:]))
    ℓlik = logpdf_particles(om, inf, ysamp)
    @test (@prob ℓlik > 0) == 0
    marginal_likelihood(ℓlik)

    mtrue = SIRModel(S₀=0.7, β=0.3)
    inf_true = solve(mtrue; save_idxs=2, saveat=1).u
    p = observe_params(om, inf)
    @test_broken begin
        ytrue = Particles(observe_dist(om, inf_true))
        marginal_divergence((:S₀,), (S₀=0.7, β=0.3), m, om; save_idxs=2, saveat=1)
    end
end

# @testset "ObservationModel Basic Tests" begin
#     s = param_sample(obs_mod)
#     @test s[2] isa Float64
#     ts_dist = obs_tspan(1:5, obs_mod, 4)
#     @test ts_dist isa Distribution
#     @test length(ts_dist) == 4
#     @test all(mean(ts_dist) .≥ (1:4 .* 100))
# end

@testset "Poisson tspan" begin
    obs_mod = PoissonTests(100.)
    ts_dist = obs_tspan(1:5, obs_mod, 4)
    @test all(mean(ts_dist) .≈ 100:100:400)
    @test all(var(ts_dist) .≈ 100:100:400)
    @test all(isapprox.(mean(rand(ts_dist, 100), dims=2), 100:100:400; atol=10))
end