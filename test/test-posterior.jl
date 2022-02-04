using DEParamDistributions
using Test
using Distributions

## Constructors
d = (1, 3, 0.5)
nl = DEParamDistributions.new_lik(d, [1/5, 0, 1]; budget=100)
@test length(nl) == 2
@test first.(Distributions.params.(nl.v)) == [10., 50.]