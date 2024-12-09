module power_management

import POMDPs
using POMDPs: POMDP
# using POMDPTools: Deterministic, Uniform, SparseCat

using Random
using Distributions
using Statistics

Random.seed!(42) # TODO: check if this actually works?


Base.@kwdef mutable struct PowerManagementPOMDP <: POMDP{Int, Int, Int} # NEED TO EDIT
    num_inst::Int
    priority_arr::Vector{Int} = rand(1:10, num_inst)
    inst_battery_usage::Vector{Int} = rand(1:5, num_inst) # want this to be higher than the battery we have available
    battery_cp::Int = 10 # battery capacity

    # failure modes # >>     failure::Bool = False
    # inst_battery_variance::Vector{Int} = 0 #rand(0:0, num_inst)

    # Reward Constants
    exceeded_capacity::Float64 = 1000

    # Discount
    discount_factor::Float64 = 0.9
end




POMDPs.states(m::PowerManagementPOMDP) = 0:m.battery_cp
POMDPs.initialstate(m::PowerManagementPOMDP) = 0
POMDPs.stateindex(m::PowerManagementPOMDP, s::Int) = s+1
POMDPs.discount(m::PowerManagementPOMDP) = m.discount_factor
POMDPs.actions(m::PowerManagementPOMDP) = 0:2^m.num_inst
POMDPs.actionindex(m::PowerManagementPOMDP,a::Int) = a+1
POMDPs.observations(m::PowerManagementPOMDP) = 0:m.battery_cp #s.battery_usage
POMDPs.obsindex(m::PowerManagementPOMDP, s::Int) = s+1
POMDPs.support(m::PowerManagementPOMDP) = Discrete(0:m.battery_cp )


function POMDPs.transition(m::PowerManagementPOMDP, s, a)
    s = 0 # reset at the beginning each time
    action_vec = digits(a,base=2)

    # battery usage and battery collection
    for (a, ind) in zip(action_vec, 1:length(action_vec))
        s -= a*m.inst_battery_usage[ind]#rand(Normal(m.inst_battery_usage[ind], m.inst_battery_variance[ind])) # is it bad to put in randomness in how much we are using battery?
    end

    # # if we take in invalid action we just lost a step?
    # if s < m.battery_cp
    #     # if valid, properly add data collection rates for each instrument
    #     s.total_collected[ind] += dot(m.inst_collection_rate,action_vec) # could also add randomness here
    # end

    return s
end


function POMDPs.reward(m::PowerManagementPOMDP, s, a)

    reward = 0.0
    action_vec = digits(a,base=2)

    # negative rewards for idle times
    reward -= dot(m.priority_arr,action_vec)

    # negative rewards for battery used past capacity assigning too many instruments
    if s > m.battery_cp
        reward -= m.exceeded_capacity
    end

    return reward
end

end # for module