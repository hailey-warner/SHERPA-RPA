module power_management

import POMDPs
using POMDPs: POMDP
# using POMDPTools: Deterministic, Uniform, SparseCat

using Random
using Distributions
using Statistics

# OVERALL general questions:
# 1) Do we need total data collected as a variable to discern between how the average is between collecting information from different instruments
#       1.1) Should this only be data collected at each timestep if so? or is a cumulative measure like this something good?
# 2) Is it a double penalty if we have two both the idle time and the data collected metrics?
# 3) How do we make rewards dependent on next state? Would it be better use the next state of the battery instead and add a power generation component?
# TODO: (3) Would it be better use the next state of the battery instead and add a power generation component?
# TODO: (2) is this a double penalty? 
# TODO: also should this be total data colleted? or just collected at this step?
# could we have both?

# reward dependent on next state example:
# https://juliapomdp.github.io/POMDPs.jl/v0.7/faq/

Random.seed!(42) # TODO: check if this actually works?


struct State
    battery_usage::Float64

    # TODO: (1) Do we need this?
    total_collected::Vector{Float64}
end

# struct Action
#     action_list::Vector{Int} # 0 or 1? # TODO: Is this a valid way of setting this up?
# end

# Action is now a integer between 0 and 2^instruments-1 which we convert into binary

Base.@kwdef mutable struct PowerManagementPOMDP <: POMDP{State, Int, State} # NEED TO EDIT
    num_inst::Int
    priority_arr::Vector{Int} = rand(1:10, num_inst)
    inst_battery_usage::Vector{Float64} = rand(1.0:30.0, num_inst) # want this to be higher than the battery we have available
    battery_cp::Float64 = 100.0 # battery capacity

    # TODO: (1) data collection rate, do we need this?
    inst_collection_rate::Vector{Float64} = rand(0.0:5.0, num_inst) # change rate of data collection of diff instruments?

    # failure modes # >>     failure::Bool = False
    variance_battery::Float64 = 0 # play around with this value?
    inst_battery_variance::Vector{Float64} = rand(0.0:0.0, num_inst)

    # Reward Constants
    exceeded_capacity::Float64 = 1000
    data_collected::Float64 = 100
    std_data::Float64 = 50

    # Discount
    discount_factor::Float64 = 0.9


end
function POMDPs.transition(m::PowerManagementPOMDP, s, a)
    s.battery_usage = 0.0 # reset at the beginning each time
    action_vec = digits(a,base=2)
    # battery usage and battery collection
    for (a, ind) in zip(action_vec, 1:length(action_vec))
        s.battery_usage -= a*Normal(m.inst_battery_usage[ind], m.inst_battery_variance[ind]) # is it bad to put in randomness in how much we are using battery?
    end

    # if we take in invalid action we just lost a step?
    if s.battery_usage < m.battery_cp
        # if valid, properly add data collection rates for each instrument
        s.total_collected[ind] += dot(m.inst_collection_rate,action_vec) # could also add randomness here
    end

    return State(s.battery_usage,s.total_collected)
end

POMDPs.pdf = 
function POMDPs.reward(m::PowerManagementPOMDP, s, a, sp)

    reward = 0.0
    action_vec = digits(a,base=2)
    # negative rewards for idle times
    reward -= dot(m.priority_arr,action_vec)

    # negative rewards for battery used past capacity assigning too many instruments
    if s.battery_usage > m.battery_cp
        reward -= m.exceeded_capacity
    end

    # for data collected, several options for rewards?    
    reward -= dot(m.priority_arr,s.total_collected)*m.data_collected 
    reward -= std(s.total_collected)*m.std_data
    
    return reward
end


function POMDPs.observation(m::PowerManagementPOMDP,s,a)
    # battery_dist = Normal(s.battery, m.variance_battery)

    # return rand(battery_dist)
    return State(s.battery_usage,s.total_collected)
end

function POMDPs.states(m::PowerManagementPOMDP)
    return (battery_usage_range = (0.0, m.battery_cp), total_collected_dim = 3)
end

function POMDPs.initialstate(m::PowerManagementPOMDP)
    battery_usage = 0  # Starting battery usage (example)
    total_collected = [0.0, 0.0, 0.0]  # Adjust based on your context
    return State(battery_usage, total_collected)
end

POMDPs.stateindex(m::PowerManagementPOMDP, s::State) = Int64(s) + 1
POMDPs.discount(m::PowerManagementPOMDP) = m.discount_factor

# things to plot at the end:
# TODO: the histogram of how often we called each instrument?
# TODO: the total data collected?
# TODO: some other things

end