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
struct PowerManagementPOMDP <: POMDP{Int, Int, Int} 
    
    # instrument specific
    num_inst::Int
    priority_arr::Vector{Int}
    inst_battery_usage::Vector{Int}

    # Battery capacity
    battery_cp::Int
    deviations::Int

    # Rewards
    exceeded_capacity::Float64
    discount_factor::Float64
end

# # Custom constructor to handle dynamic initialization
function PowerManagementPOMDP(;
    num_inst::Int,
    battery_cp::Int = 100, 
    exceeded_capacity::Float64 = 100.0, # penalty for reward
    discount_factor::Float64 = 0.9,
    deviations::Int = 5
)
    priority_arr = rand(1:10, num_inst)
    inst_battery_usage = rand(1:50.0, num_inst)
    return PowerManagementPOMDP(
        num_inst, priority_arr, inst_battery_usage, battery_cp, deviations, exceeded_capacity, discount_factor, 
    )
end


POMDPs.states(m::PowerManagementPOMDP) = 0:m.battery_cp
POMDPs.actions(m::PowerManagementPOMDP) = 0:(2^m.num_inst)-1
POMDPs.observations(m::PowerManagementPOMDP) = 0:m.battery_cp # same as states

POMDPs.discount(m::PowerManagementPOMDP) = m.discount_factor

POMDPs.stateindex(m::PowerManagementPOMDP, s::Int) = s+1
POMDPs.actionindex(m::PowerManagementPOMDP,a::Int) = a+1
POMDPs.obsindex(m::PowerManagementPOMDP, s::Int) = s+1

POMDPs.initialstate(m::PowerManagementPOMDP) = Uniform(0:m.battery_cp) # gives back a distribution between 0 and 100

# Function to get a binary representation with a fixed number of digits
function to_fixed_binary(n::Int, num_digits::Int)
    binary_digits = digits(n, base =2)  # Convert to binary
    padded_binary =  vcat(zeros(Int, num_digits - length(binary_digits)), binary_digits)  # Pad with leading zeros if needed
    return padded_binary
end

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

function POMDPs.transition(m::PowerManagementPOMDP, s, a)

    action_vec = to_fixed_binary(a, m.num_inst)

    sp = s+50#rand(Uniform(0: m.battery_cp))
    if sp > 100
        sp = 100
    end

    # battery usage and battery collection
    for (a, ind) in zip(action_vec, 1:length(action_vec))
        sp -= a*m.inst_battery_usage[ind] #rand(Normal(m.inst_battery_usage[ind], m.inst_battery_variance[ind])) # is it bad to put in randomness in how much we are using battery?
        if sp < 0
            sp = 0
            break
        end
    end

    return Deterministic(sp)
end

function POMDPs.observation(m::PowerManagementPOMDP, a, sp)


    # TURNING ON OR OFF DISTRIBUTIONS
    if true
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

    # negative rewards for idle times
    action_vec = to_fixed_binary(a, m.num_inst)
    flipped_action = map(x -> 1 - x, action_vec)
    reward -= dot(m.priority_arr,transpose(flipped_action))
    print(reward)

    # negative rewards for battery used past capacity assigning too many instruments
    if s == 0
        reward -= m.exceeded_capacity
    end

    return reward
end

# end # for module
