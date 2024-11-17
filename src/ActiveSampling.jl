module ActiveSampling

using POMDPs
using Parameters
using Random
using StaticArrays

const NOISE_LEVEL = 0.1 #camera/sensor noise
export CreateSamplePOMDP
export SamplePOMDP
export State

struct State
    pos::SVector{2, Int}        # robotic arm's position
    full::Bool                  # whether scoop is full or empty
    collected::Int              # number of samples collected (0-3)
    belief_map::Array{Bool, 2}  # belief map of good and bad sample locations
end

@with_kw struct SamplePOMDP <: POMDP{State, Int, Int}
    map_size::Tuple{Int, Int} = (3, 3)
    init_pos::SVector{2, Int} = (2, 2)

    true_map::Array{Bool, 2} = rand(Bool, map_size...)  # ground truth quality map (1 = good, 0 = bad)
    #belief_map::Array{Bool, 2} = true_map .⊻ (rand(map_size) .< NOISE_LEVEL) # belief map (1 = good, 0 = bad)
    #belief_map::Array{Bool, 2} = [Beta(1.0, 1.0) for i in 1:n, j in 1:m] # belief map (Beta dist.)

    # Reward
    move_penalty::Float64       = -1.
    scoop_penalty::Float64      = -10.
    accept_good_reward::Float64 = 10e3
    accept_bad_penalty::Float64 = -10e5
    reject_good_reward::Float64 = -10e2
    reject_bad_penalty::Float64 = -10.

    # Misc.
    terminal_state::State = State((-1,-1), false, 3, Matrix{Bool}(undef, map_size...))
    discount_factor::Float64 = 0.95
end

function CreateSamplePOMDP()
    return SamplePOMDP()
end

POMDPs.isterminal(pomdp::SamplePOMDP, s::State) = s.collected == pomdp.terminal_state.collected # finished collecting?
POMDPs.discount(pomdp::SamplePOMDP) = pomdp.discount_factor

include("states.jl")
include("actions.jl")
include("transition.jl")
include("observations.jl")
include("reward.jl")

end