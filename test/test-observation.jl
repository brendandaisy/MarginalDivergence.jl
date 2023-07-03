using Revise
using Test
using MarginalDivergence
using MonteCarloMeasurements, Distributions

# Implement a sequence of biased coins as an observation model
struct Coins <: AbstractObservationModel{Nothing} end
    
function MarginalDivergence.logpdf_particles(::Coins, μ, data)
    sum(t->t[2]*log(t[1]) + (1 - t[2])*log(1-t[1]), zip(μ, data))
end

MarginalDivergence.observe_dist(::Coins, p) = Bernoulli.(p)

@testset "ObervationModel interface" begin
    ptrue = [0, 0.5]
    om = Coins()
    μ = outer_product(fill(Uniform(0, 1), 2), 10_000)

    # check observe_dist with particle p uses mean of p with warning
    y = observe_dist(om, ptrue)
    @test y[1] isa Distribution
    ℓpdf = log_likelihood(om, μ, [0, 0])
    @test ℓpdf isa Particles
    @test all(ℓpdf.particles .< 0)
    @test joint_observe_dist(om, ptrue) isa Distribution

    yrep = observe_dist(om, ptrue, 3)
    @test size(yrep) == (2, 3)
    ℓpdf_rep = log_likelihood(om, μ, zeros(2, 3))
    @test (@prob ℓpdf_rep <= ℓpdf) == 1.
end

@testset "Testing behavior of the provided observation models" begin
    # PoissonRate with deterministic μ
    μ = [1., 2, 3]
    om_pois = PoissonRate(100)
    y = joint_observe_dist(om_pois, μ)
    data = rand(y)
    @test mean(y) == 100μ
    # test logpdf_particles without/with multiple replications
    @test log_likelihood(om_pois, μ, data) == logpdf(y, data)
    @test_throws ["no method", "logfactorial"] log_likelihood(om_pois, μ, [100., 200, 300])
    yrep = joint_observe_dist(om_pois, μ, 3)
    data = rand(yrep) # sample a 3x3 matrix
    @test log_likelihood(om_pois, μ, data) ≈ logpdf(yrep, data)
    # test random μ and rate η works as expected
    μ = max.(μ .± 0.3, 0.1)
    om_pois = PoissonRate(100..200)
    data = [100, 200, 300]
    raw_dists = map(1:2000) do i
        product_distribution([Poisson.(μt.particles[i] * om_pois.η.particles[i]) for μt in μ])
    end
    @test all(log_likelihood(om_pois, μ, data).particles .≈ [logpdf(d, data) for d in raw_dists])
    # test NConstVar
    μ = [100., 200.]
    om_norm = NConstVar(5)
    y = joint_observe_dist(om_norm, μ)
    data = rand(y)
    @test mean(y) == μ
    @test log_likelihood(om_norm, μ, data) ≈ logpdf(y, data)
end

@testset "Latent parameter types are propogated through observation" begin
    m = SIRModel{Float32}(S₀=0f0..1f0, β=Particles(TruncatedNormal(0.3f0, 0.1f0, 0f0, Inf32)))
    inf = solve(m; save_idxs=2, saveat=1:3).u
    om_pois = PoissonRate(100)

    @test log_likelihood(om_pois, inf, [1, 10, 20]) isa Particles{Float32}
    @test log_likelihood(PoissonRate(true), inf, [1, 10, 20]) isa Particles{Float32}
    @test log_likelihood(PoissonRate(100.0), inf, [1, 10, 20]) isa Particles{Float64} # here the Float64 takes precedence when *inf
    @test log_likelihood(PoissonRate(100f0), inf, [1 10 20; 5 7 18]) isa Particles{Float32}

    @test eltype(observe_dist(om_pois, particles_index(inf, 3))) <: Poisson{Float32}

    om_norm = NConstVar(0.1f0)
    @test log_likelihood(om_norm, inf, [1, 10, 20]) isa Particles{Float32}
    @test log_likelihood(NConstVar(0.1), inf, [1, 10, 20]) isa Particles{Float64}
    @test log_likelihood(NConstVar(1), inf, [1, 10, 20]) isa Particles{Float64} # because log(Int) isa Float64
    @test log_likelihood(om_norm, inf, [1, 10.0, 20]) isa Particles{Float64} # Real are in data domain, unlike Poisson, so Float64 can also take precedence there
end