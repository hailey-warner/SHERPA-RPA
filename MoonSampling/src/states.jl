# States

function Base.length(mdp::ExtractionMDP) # size of state space
    return (mdp.map_size[1] * mdp.map_size[2])^3 * 2 * 4
end

POMDPs.states(mdp::ExtractionMDP) = [
    ExtractionState([i,j], full, collected, [a,b], [c,d])
    for i in 1:mdp.map_size[1]
    for j in 1:mdp.map_size[2]
    for full in [false, true]
    for collected in 0:3
    for a in 1:mdp.map_size[1]
    for b in 1:mdp.map_size[2] 
    for c in 1:mdp.map_size[1]
    for d in 1:mdp.map_size[2]
]

POMDPs.initialstate(mdp::ExtractionMDP) = Deterministic(ExtractionState([1, 1], false, 0, [0,0], [1,1]))

# transform ExtractionState into vector
function POMDPs.convert_s(T::Type{<:AbstractArray}, s::ExtractionState, mdp::ExtractionMDP)
    return convert(T, vcat(s.pos, s.full, s.collected, s.sample_one, s.sample_two))
end

# transform vector to ExtractionState
function POMDPs.convert_s(T::Type{ExtractionState}, v::AbstractArray, mdp::ExtractionMDP)
    return ExtractionState([v[1], v[2]], Bool(v[3]), Int(v[4]), [v[5], v[6]], [v[7], v[8]])
end

function POMDPs.isterminal(mdp::ExtractionMDP, s::ExtractionState)
    return s.collected >= 3  # Terminate when we have 3 samples onboard
end

function POMDPs.stateindex(mdp::ExtractionMDP, s::ExtractionState)
    i, j = s.pos
    a, b = s.sample_one
    c, d = s.sample_two
    n_cols = mdp.map_size[2]
    
    # calculate base indices for each component
    pos_idx = (i-1) * n_cols + (j-1)
    full_idx = s.full ? 1 : 0
    collected_idx = s.collected
    sample_one_idx = (a-1) * n_cols + (b-1)
    sample_two_idx = (c-1) * n_cols + (d-1)
    
    return 1 + pos_idx + 
           full_idx * (mdp.map_size[1] * mdp.map_size[2]) + 
           collected_idx * (mdp.map_size[1] * mdp.map_size[2] * 2) +
           sample_one_idx * (mdp.map_size[1] * mdp.map_size[2] * 2 * 4) +
           sample_two_idx * (mdp.map_size[1] * mdp.map_size[2] * 2 * 4 * (mdp.map_size[1] * mdp.map_size[2]))
end