module ActiveSampling

using POMDPs
using Parameters
using Random

struct State
    pos::Vector{2, Int}
    map::Array{2, Int}
    occupied::bool
    collected::Int
end

@with_kw struct SamplePOMDP <: POMDP{State, Int, Int} # {S type, A type, O type}
    map_size::Tuple{Int, Int} = (7, 7)
    init_pos::Vector{2, Int} = (4, 4)
    init_map::Array{2, Int} = rand()
    #terminal_state

    # Reward
    move_penalty::Float64 = -1.
    scoop_penalty::Float64 = -10.
    accept_valid_reward::Float64 = 10e3
    accept_invalid_penality::Float64 = -10e5
    reject_valid_reward::Float64 = -10e2
    reject_invalid_penality::Float64 = -10.

    # Misc.
    discount::Float64 = 0.9
    #indices
    #sensor_efficiency
end

function GenerateMap()
end

function GeneratePOMDP()
end

include("states.jl")
include("actions.jl")
include("transition.jl")
include("observations.jl")
include("reward.jl")

end