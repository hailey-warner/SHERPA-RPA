# Actions

# [accept, reject, scoop11, scoop12, ..., scoopmn]
POMDPs.actions(mdp::ExtractionMDP) = collect(1:(mdp.map_size[1]*mdp.map_size[2]+2))

# State-Dependent Action Space (for MCTS)
#https://github.com/JuliaPOMDP/POMDPs.jl/discussions/353
function POMDPs.actions(mdp::ExtractionMDP, s::ExtractionState)
    if s.full == true
        return [1, 2] # [accept, reject]
    else
        return collect(3:(mdp.map_size[1]*mdp.map_size[2])+2)
    end
end

POMDPs.actionindex(mdp::ExtractionMDP, a::Int) = a