using Revise
using Test
using DEParamDistributions
using Distributions

obs_mod = PoissonBiasMult(100., Uniform(1, 5))

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