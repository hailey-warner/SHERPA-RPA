module ActiveSampling

using POMDPs
using Parameters
using Random

struct State
    pos::Vector{2, Int}
    qual_map::Array{2, Int}
    conf_map::Array{2, Int}
    occupied::bool
    collected::Int
end

@with_kw struct SampleMDP <: MDP{State, Int, Int} # {S type, A type, O type}
    map_size::Tuple{Int, Int} = (7, 7)
    init_pos::Vector{2, Int} = (4, 4)
    true_map::Array{2, Int} = rand(-10:10, map_size...) # true quality map (unknown)
    qual_map:: Array{2, Int} = zeros(Int, map_size...) # uniform prior (true + noise?)
    conf_map:: Array{2, Int} = zeros(Int, map_size...) # uniform prior

    # Reward
    move_penalty::Float64 = -1.
    scoop_penalty::Float64 = -10.
    accept_good_reward::Float64 = 10e3
    accept_bad_penalty::Float64 = -10e5
    reject_good_reward::Float64 = -10e2
    reject_bad_penalty::Float64 = -10.

    # Misc.
    discount::Float64 = 0.95
    #indices
end

function GenerateMap()
    # TODO: uniform prior (or true_map + noise?)
end

function GeneratePOMDP()
end

# TODO: terminate when collected = 3

include("states.jl")
include("actions.jl")
include("transition.jl")
include("observations.jl")
include("reward.jl")

end