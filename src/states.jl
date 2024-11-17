POMDPs.states(pomdp::SamplePOMDP) = pomdp # TODO: is the state space the pomdp itself? GPT says this should be an iterator?

# state space = belief map * robot arm pos # samples collected * full + terminal
Base.length(pomdp::SamplePOMDP) = 2^(pomdp.map_size[1]*pomdp.map_size[2]) * (pomdp.map_size[1]*pomdp.map_size[2]) * 3 * 2 + 1



# TODO: write these functions
# what to use in place of K?
function POMDPs.stateindex(pomdp::SamplePOMDP, s::State)
    if isterminal(pomdp, s)
        return length(pomdp)
    end
    return s.pos[1] + pomdp.indices[1] * (s.pos[2]-1) + dot(view(pomdp.indices, 2:(K+1)), s.rocks)
end

function state_from_index(pomdp::SamplePOMDP, si::Int)
    if si == length(pomdp)
        return pomdp.terminal_state
    end
    rocks_dim = @SVector fill(2, K)
    nx, ny = pomdp.map_size
    s = CartesianIndices((nx, ny, rocks_dim...))[si]
    pos = RSPos(s[1], s[2])
    #rocks = SVector{K, Bool}(s.I[3:(K+2)] .- 1)
    return RSState{K}(pos, rocks)
end

function Base.iterate(pomdp::SamplePOMDP, i::Int=1)
    if i > length(pomdp)
        return nothing
    end
    s = state_from_index(pomdp, i)
    return (s, i+1)
end

function POMDPs.initialstate(pomdp::SamplePOMDP)
    belief_map = pomdp.true_map .⊻ (rand(pomdp.map_size) .< NOISE_LEVEL)
    probs = 
    states = 
    return SparseCat(states, probs)
end