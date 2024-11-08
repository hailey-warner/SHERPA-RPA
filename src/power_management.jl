# module power_management

import POMDPs
using POMDPs: POMDP
using POMDPTools: Deterministic, Uniform, SparseCat

# using Distributions
using Statistics # std
using Random # rand, and random seed
using LinearAlgebra


Random.seed!(42) # TODO: check if this actually works?

# Define mutable struct without default values, dynamic fields
struct PowerManagementPOMDP <: POMDP{Int, Int, Int} 
    
    # instrument specific
    num_inst::Int
    priority_arr::Vector{Int}
    inst_battery_usage::Vector{Int}

    # Batteru capacity
    battery_cp::Int

    # Rewards
    exceeded_capacity::Float64
    discount_factor::Float64
end

# # Custom constructor to handle dynamic initialization
function PowerManagementPOMDP(;
    num_inst::Int,
    battery_cp::Int = 100, 
    exceeded_capacity::Float64 = 10.0, 
    discount_factor::Float64 = 0.9
)
    priority_arr = rand(1:10, num_inst)
    inst_battery_usage = rand(1:50, num_inst)
    return PowerManagementPOMDP(
        num_inst, priority_arr, inst_battery_usage, battery_cp, exceeded_capacity, discount_factor
    )
end


POMDPs.states(m::PowerManagementPOMDP) = 0:m.battery_cp
POMDPs.actions(m::PowerManagementPOMDP) = 0:(2^m.num_inst)-1
POMDPs.observations(m::PowerManagementPOMDP) = 0:m.battery_cp # same as states

POMDPs.discount(m::PowerManagementPOMDP) = m.discount_factor

POMDPs.stateindex(m::PowerManagementPOMDP, s::Int) = s+1
POMDPs.actionindex(m::PowerManagementPOMDP,a::Int) = a+1
POMDPs.obsindex(m::PowerManagementPOMDP, s::Int) = s+1

POMDPs.initialstate(m::PowerManagementPOMDP) = Deterministic(0)

# Function to get a binary representation with a fixed number of digits
function to_fixed_binary(n::Int, num_digits::Int)
    binary_digits = digits(n, base =2)  # Convert to binary
    return vcat(zeros(Int, num_digits - length(binary_digits)), binary_digits)  # Pad with leading zeros if needed
end

function POMDPs.transition(m::PowerManagementPOMDP, s, a)
    sp = 0 # reset at the beginning each time
    action_vec = to_fixed_binary(a, m.num_inst)


    # battery usage and battery collection
    for (a, ind) in zip(action_vec, 1:length(action_vec))
        sp += a*m.inst_battery_usage[ind] #rand(Normal(m.inst_battery_usage[ind], m.inst_battery_variance[ind])) # is it bad to put in randomness in how much we are using battery?
    end

    return Deterministic(sp)
end

function POMDPs.observation(m::PowerManagementPOMDP, a, sp)
    return Deterministic(sp)
end

function POMDPs.reward(m::PowerManagementPOMDP, s, a)

    reward = 0.0

    # negative rewards for idle times
    action_vec = to_fixed_binary(a, m.num_inst)
    reward -= dot(m.priority_arr,transpose(action_vec))

    # negative rewards for battery used past capacity assigning too many instruments
    if s > m.battery_cp
        reward -= m.exceeded_capacity
    end

    return reward
end

# end # for module