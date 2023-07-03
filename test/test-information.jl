using OrdinaryDiffEq

@testset "Compare RMD with some analytical results" begin
    om = Coins()
    μ = outer_product(fill(Uniform(0, 1), 2), 10_000)
    μtrue = [0, 0.5]
    ytrue = observe_dist(om, μtrue)

    md = marginal_divergence(ytrue, μtrue, μ, om)
    # ytrue is [0, 0] or [0,1], both of which have P(y)=1/2, so 
    @test md ≈ (log(1/2) - log(1/4))

    # TODO 7/2: more examples! Binomial (+ compare with multiple reps of Coins?), linear regression
    # μinit = Particles(10_000, Normal(10, 0.5))
    # prob = ODEProblem((u, p, t) -> 2u, μinit, (0., 3.))
    # μ = solve(prob, Tsit5(); saveat=0:3)
end

@testset "Parametric types are propogated through information methods" begin
    m = SIRModel{Float32}(S₀=0f0..1f0, β=Particles(TruncatedNormal(0.3f0, 0.1f0, 0f0, Inf32)))
    inf = solve(m; save_idxs=2, saveat=1:3).u
    om_pois = PoissonRate(100)
    ℓ1 = log_likelihood(om_pois, inf, [1, 10, 20]) 
    ℓ2 = log_likelihood(NConstVar(0.5f0), inf, [1, 10, 20])

    @test (marginal_likelihood(ℓ1) isa Float32) & (marginal_likelihood(ℓ2) isa Float32)

    y = observe_dist(om_pois, pmean.(inf)) # use of pmean means RMD won't technically correct
    inf_cond = solve(m, (S₀=0.7f0,); save_idxs=2, saveat=1).u
    md = marginal_divergence(y, inf_cond, inf, om_pois; N=100)
    @test md isa Float32
end

@testset "SIR model and marginal divergence runs as expected" begin
    m = SIRModel(S₀=0..1, β=Particles(TruncatedNormal(0.3, 0.1, 0, Inf)))
    om = PoissonRate(100)

    md1 = marginal_divergence((:β, :S₀), (S₀=0.7, β=0.3), m, om; save_idxs=2, saveat=1, N=100)
    md2 = marginal_divergence((:S₀,), (S₀=0.7, β=0.3), m, om; save_idxs=2, saveat=1, N=100)

    @test md1 > md2

    # inf_pri = solve(m; save_idxs=2, saveat=1).u
    # inf_true = solve(m, (S₀=0.7, β=0.3); save_idxs=2, saveat=1).u
    # ytrue = observe_dist(om, inf_true)
    # inf_cond = solve(m, (S₀=0.7,); save_idxs=2, saveat=1).u

    # # adding noise to observation processes should reduce RMD
    # md3 = marginal_divergence(ytrue, inf_cond, inf_pri, om; N=1000)
    # om = PoissonRate(100..1000)
    # md4 = marginal_divergence(ytrue, inf_cond, inf_pri, om; N=1000)
    # om = PoissonRate(100..10_000)
    # md5 = marginal_divergence(ytrue, inf_cond, inf_pri, om; N=1000)

    # @test md3 > md4
    # @test md4 > md5
end