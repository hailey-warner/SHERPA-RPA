module moon

using POMDPs
using POMDPTools
using StaticArrays
using Distributions
using LinearAlgebra
using GaussianProcesses
using Random
using MCTS
using DiscreteValueIteration
using SARSOP
using Plots
Random.seed!(84)

# Structs
export
    ExtractionPOMDP,
    ExtractionState

# Discount
POMDPs.discount(mdp::ExtractionMDP) = 0.95

include("states.jl")
include("actions.jl")
include("transition.jl")
include("observations.jl")
include("reward.jl")
include("belief.jl")

end # module