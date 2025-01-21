# Actions
# Rotate + Extend

POMDPs.actions(mdp::ExtractionMDP) = collect(1:23+mdp.map_size)

function POMDPs.actions(mdp::ExtractionMDP, s::ExtractionState)
    if s.full == true
        return [1, 2] # [accept, reject]
    else
        return collect(3:23+mdp.map_size) # [scoop, -90°, -80°, ..., 80°, 90°, extend 1, extend 2, ..., extend map_size]
    end
end

POMDPs.actionindex(mdp::ExtractionMDP, a::Int) = a