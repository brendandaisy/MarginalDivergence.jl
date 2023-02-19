using Revise
using Test
using MarginalDivergence
using MonteCarloMeasurements, Distributions

@testset "Biased Coins" begin
    struct Coins <: AbstractObservationModel{Int8} end
    
    function MarginalDivergence.logpdf_particles(::Coins, x::Vector{<:Real}, data)
        sum(t->t[2]*log(t[1]) + (1 - t[2])*log(1-t[1]), zip(x, data))
    end
    MarginalDivergence.observe_dist(::Coins; p) = product_distribution(Bernoulli.(p))

    om = Coins()
    x = outer_product(fill(Uniform(0, 1), 2), 10_000)
    @test observe_params(om, x).μ === x # by default, only params are a mean μ=x, just what we need for this model!

    ℓlik = logpdf_particles(om, x, [0, 0])
    @test marginal_likelihood(ℓlik) ≈ log(1/4)

    xtrue = [0, 0.5]
    ytrue = Particles(observe_dist(om; p=xtrue))
    xpart = [0, Particles(10_000, Uniform(0, 1))]

    md = marginal_divergence(ytrue, xpart, x, om)
    md2 = marginal_divergence(ytrue, xpart, om; ℓp_of_y_precomp=ℓlik)
    # ytrue is [0, 0] or [0,1], both of which have MℓL=log(1/2), so 
    @test md ≈ (log(1/2) - log(1/4))
end

@testset "Latent parameter types are propogated through observation" begin
    m = SIRModel{Float32}(S₀=0f0..1f0, β=Particles(TruncatedNormal(0.3f0, 0.1f0, 0f0, Inf32)))
    inf = solve(m; save_idxs=2, saveat=1).u
    om = PoissonRate(100)
    par = observe_params(om, inf)
    @test eltype(par.λ) <: Particles{Float32}
    ysamp = rand(observe_dist(om; λ=vecindex(par.λ, 1)))
    @test typeof(logpdf_particles(om, inf, ysamp)) <: Particles{Float32}
    @test marginal_likelihood(logpdf_particles(om, inf, ysamp)) isa Float32

    inf2 = solve(m, (S₀=0.7f0, β=0.3f0); save_idxs=2, saveat=1).u
    ysamp = rand(observe_dist(om; λ=observe_params(om, inf2).λ))
    ℓ2 = logpdf_particles(om, inf2, ysamp)
    @test ℓ2 isa Float32
    @test marginal_likelihood(ℓ2) isa Float32
end

@testset "Parametric types are propogated through information methods" begin
    m = SIRModel{Float32}(S₀=0f0..1f0, β=Particles(TruncatedNormal(0.3f0, 0.1f0, 0f0, Inf32)))
    inf = solve(m; save_idxs=2, saveat=1).u
    om = PoissonRate(100)
    
    md = marginal_divergence((:β, :S₀), (S₀=0.7f0, β=0.3f0), m, om; save_idxs=2, saveat=1)
    @test md isa Float32
end

@testset "Sanity checks with SIR model and marginal divergence" begin
    m = SIRModel(S₀=0..1, β=Particles(TruncatedNormal(0.3, 0.1, 0, Inf)))
    inf = solve(m; save_idxs=2, saveat=1).u
    om = PoissonRate(100)
    λsamp = observe_params(om, inf, 1:2)
    ysamp = rand(observe_dist(om; λ=100*Matrix(inf)[1,1:2]), 100_000)
    μy = mean.(eachrow(ysamp))
    @test isapprox(μy, vecindex(λsamp.λ, 3); atol=0.05)

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
    om = PoissonRate(Particles(DiscreteUniform(100, 200)))

    p = observe_params(om, inf)
    ysamp = rand(observe_dist(om, Matrix(inf)[1,:]))
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