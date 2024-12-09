# module power_management
import POMDPs
using POMDPs: POMDP
using POMDPTools #: Deterministic, Uniform, SparseCat
using Distributions: Normal

# using Distributions
using Statistics # std
using Random # rand, and random seed
using LinearAlgebra
using SparseArrays



Random.seed!(42) # TODO: check if this actually works?

# Define mutable struct without default values, dynamic fields
struct PowerManagementPOMDP  <: POMDP{Int, Int, Int} 
    
    # instrument specific
    num_inst::Int
    priority_arr::Vector{Int}
    inst_battery_usage::Vector{Int}

    # Battery capacity
    battery_cp::Int
    deviations::Int
    charging::Int

    # idle timesteps
    idle_time_max::Int
    idle_times::Vector{Int} # mutable? maybe include this instead in the pomdp state

    # Rewards
    exceeded_capacity::Float64
    discount_factor::Float64
    idle_time_max_penalty::Float64

end

# # Custom constructor to handle dynamic initialization
function PowerManagementPOMDP(;
    num_inst::Int,
    priority_arr::Vector{Int}, #rand(1:10, num_inst)
    inst_battery_usage::Vector{Int},#rand(1:battery_cp, num_inst)
    idle_time_max::Int= 10,
    charging::Int = 100,
    battery_cp::Int = 100, 
    exceeded_capacity::Float64 = 100.0, # penalty for reward
    discount_factor::Float64 = 0.9,
    idle_time_max_penalty::Float64 = 10.0, 
    deviations::Int = 5,
)
    # priority_arr = [1,10,10,1,1] #rand(1:10, num_inst)
    # inst_battery_usage = [30, 30, 30, 30, 30]#rand(1:battery_cp, num_inst)
    idle_times = zeros(Int, num_inst)
    return PowerManagementPOMDP(
        num_inst, priority_arr, inst_battery_usage, battery_cp, deviations, charging, idle_time_max, idle_times, exceeded_capacity, discount_factor, idle_time_max_penalty,
    )
end


POMDPs.states(m::PowerManagementPOMDP) = 0:(m.battery_cp*((2^m.num_inst)))+(2^m.num_inst)-1

# Function to get a binary representation with a fixed number of digits
function to_fixed_binary(n::Int, num_digits::Int)
    binary_digits = digits(n, base =2)  # Convert to binary
    padded_binary = vcat(binary_digits,zeros(Int, num_digits - length(binary_digits)))  # Pad with leading zeros if needed
    return padded_binary
end

POMDPs.actions(m::PowerManagementPOMDP) = 0:(2^m.num_inst)-1

function POMDPs.actions(m::PowerManagementPOMDP, s) 
    possible_actions = Int[]
    for a in 0:(2^m.num_inst)-1
        action_vec = reverse(to_fixed_binary(a, m.num_inst))
        if (dot(m.inst_battery_usage,action_vec) < s)
            push!(possible_actions, a)
        end
    end
    return possible_actions
end



POMDPs.observations(m::PowerManagementPOMDP) = 0:(m.battery_cp*((2^m.num_inst)))+(2^m.num_inst)-1 # same as states

POMDPs.discount(m::PowerManagementPOMDP) = m.discount_factor

POMDPs.stateindex(m::PowerManagementPOMDP, s::Int) = s+1 
POMDPs.actionindex(m::PowerManagementPOMDP,a::Int) = a+1
POMDPs.obsindex(m::PowerManagementPOMDP, s::Int) = s+1

POMDPs.initialstate(m::PowerManagementPOMDP) = Uniform(0:m.battery_cp) # gives back a distribution between 0 and 100


function create_sparsecat_from_normal(mean::Float64, std::Float64, num_bins::Int, bins_range::Tuple{Float64, Float64})
    # Step 1: Define the normal distribution
    normal_dist = Normal(mean, std)

    # Step 2: Define the bin edges and calculate frequencies
    bin_edges = range(bins_range[1], bins_range[2], length=num_bins+1)
    frequencies = zeros(Int, num_bins)

    for x in rand(normal_dist, 10_000)  # Sample data for binning
        if x >= bins_range[1] && x <= bins_range[2]
            bin_index = findfirst(bin_edges .>= x) - 1
            frequencies[bin_index] += 1
        end
    end

    # Step 3: Normalize the frequencies into probabilities
    probabilities = frequencies / sum(frequencies)

    # Step 4: Create a sparse categorical distribution
    sparse_probs = sparse(probabilities)  # Sparse representation
    sparse_cat = SparseCat(1:num_bins, sparse_probs)

    return sparse_cat
end

function state_decompose(idle_time_max::Int, index::Int)
    battery_level = div(index, idle_time_max)
    max_idle = index % idle_time_max
    return battery_level, max_idle
end

function state_toindex(battery_level::Int, prev_state::Int, action_space::Int, battery_cp::Int)
    index = battery_level * action_space + prev_state
    # if index > battery_cp*idle_time_max
    #     index = battery_cp* idle_time_max
    # end
    return index
end

function POMDPs.transition(m::PowerManagementPOMDP, s, a)

    action_vec = (to_fixed_binary(a, m.num_inst)) # reverse

    battery_level, prev_state = state_decompose((2^m.num_inst), s)

    battery_level = battery_level+m.charging #rand(Uniform(0: m.battery_cp))
    if battery_level > m.battery_cp
        battery_level = m.battery_cp
    end

    battery_notdead = true
    # battery usage and battery collection
    temp = battery_level
    for (a, ind) in zip(action_vec, 1:length(action_vec))
        temp -= a*m.inst_battery_usage[ind] #rand(Normal(m.inst_battery_usage[ind], m.inst_battery_variance[ind])) # is it bad to put in randomness in how much we are using battery?
        # if a == 0
        #     m.idle_times[ind] += 1
        # else
        #     m.idle_times[ind] = 0
        # end

        if temp < 0
            battery_level = 0
            battery_notdead = false
            break
        else
            battery_level = temp
        end
    end

    if battery_notdead
        # check idles
        for (a, ind) in zip(action_vec, 1:length(action_vec))
            if a == 0
                m.idle_times[ind] += 1
            else
                m.idle_times[ind] = 0
            end
        end
    end
    
    prev_state = a #maximum(m.idle_times)

    # if max_idle > m.idle_time_max-1
    #     max_idle = m.idle_time_max-1
    # end
    sp = state_toindex(Int(battery_level), prev_state, (2^m.num_inst), m.battery_cp)

    return Deterministic(sp)
end

function POMDPs.observation(m::PowerManagementPOMDP, a, sp)
    # battery_level, max_idle = state_decompose(m.idle_time_max, sp)

    # battery_level = battery_level+m.charging #rand(Uniform(0: m.battery_cp))
    # if battery_level > m.battery_cp
    #     battery_level = m.battery_cp
    # end

    # sp = state_toindex(Int(battery_level), max_idle, m.idle_time_max, m.battery_cp)

    # return Deterministic(sp)
    # TURNING ON OR OFF DISTRIBUTIONS
    # NEED TO EDIT TO DO APPROPRIATE SPARSECAT WITH THIS NEW THING
    if false
        # Example Usage
        mean = sp
        std = m.deviations
        num_bins = 10
        bins_range = (sp-5,sp+5)

        # Step 1: Define the normal distribution
        normal_dist = Normal(mean, std)

        # Step 2: Define the bin edges and calculate frequencies
        bin_edges = range(bins_range[1], bins_range[2], length=num_bins+1)
        frequencies = zeros(Int, num_bins)

        for x in rand(normal_dist, 10_000)  # Sample data for binning
            if x >= bins_range[1] && x <= bins_range[2]
                bin_index = findfirst(bin_edges .>= x) - 1
                frequencies[bin_index] += 1
            end
        end

        # Step 3: Normalize the frequencies into probabilities
        probabilities = frequencies / sum(frequencies)

        # Step 4: Create a sparse categorical distribution
        sparse_probs = sparse(probabilities)  # Sparse representation

        return SparseCat(1:num_bins, sparse_probs) 
    else
        return Deterministic(sp) #Normal(sp,m.deviations)
    end
end

function POMDPs.reward(m::PowerManagementPOMDP, s, a)

    reward = 0.0

    battery_level, prev_action = state_decompose((2^m.num_inst), s)

    if battery_level == 0
        reward -= 1000000#m.exceeded_capacity
    else
    # negative rewards for idle times
        action_vec = (to_fixed_binary(a, m.num_inst)) #reverse
        flipped_action = map(x -> 1 - x, action_vec) 
        reward -= dot(m.priority_arr,flipped_action) #*max_idle

        if prev_action == a
            reward -= 1
        end
    end


    return reward
end

# end # for module
