module MoonSampling

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
    ExtractionMDP,
    ExtractionState

struct ExtractionState
    pos::Vector{Int}
    full::Bool
    collected::Int
    sample_one::Vector{Int}
    sample_two::Vector{Int}
end

mutable struct ExtractionMDP <: MDP{ExtractionState, Int}
    map_size::SVector{2, Int}
    truth_map::Array{Bool, 2}
end

# Discount
POMDPs.discount(mdp::ExtractionMDP) = 0.95

include("states.jl")
include("actions.jl")
include("transition.jl")
include("observations.jl")
include("reward.jl")
include("belief.jl")

end # module