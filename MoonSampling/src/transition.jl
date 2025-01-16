# Transitions

"""
function POMDPs.transition(mdp::ExtractionMDP, s::ExtractionState, a::Int)
    if isterminal(mdp, s)
        return Deterministic(s)
    end
    if a in 3:mdp.map_size[1]*mdp.map_size[2]+2 && !s.full
        pos_idx = a - 2
        i = (pos_idx-1) ÷ mdp.map_size[2] + 1
        j = (pos_idx-1) % mdp.map_size[2] + 1
        if s.sample_one == [1,1] # 90% chance of successfully scooping
            return SparseCat([ExtractionState([i,j], true, s.collected, s.pos, s.sample_two), ExtractionState([i,j], false, s.collected, s.sample_one, s.sample_two)], [0.9,0.1])
        elseif s.sample_two == [1,1]
            return SparseCat([ExtractionState([i,j], true, s.collected, s.sample_one, s.pos), ExtractionState([i,j], false, s.collected, s.sample_one, s.sample_two)], [0.9,0.1])
        else # third sample
            return SparseCat([ExtractionState([i,j], true, s.collected, s.sample_one, s.sample_two), ExtractionState([i,j], false, s.collected, s.sample_one, s.sample_two)], [0.9,0.1])
        end
    end
    if a == 1 && s.full # accept
        return Deterministic(ExtractionState(s.pos, false, s.collected + 1, s.sample_one, s.sample_two))
    end
    if a == 2 && s.full # reject
        return Deterministic(ExtractionState(s.pos, false, s.collected, s.sample_one, s.sample_two))
    end
    return Deterministic(s) # default (do nothing)
end

"""

# Rotate + Extend

function POMDPs.transition(mdp::ExtractionMDP, s::ExtractionState, a::Int)
    if isterminal(mdp, s)
        return Deterministic(s)
    end
    if a in 3:mdp.map_size[1]*mdp.map_size[2]+2 && !s.full
        pos_idx = a - 2
        i = (pos_idx-1) ÷ mdp.map_size[2] + 1
        j = (pos_idx-1) % mdp.map_size[2] + 1
        if s.sample_one == [1,1] # not yet collected
            return SparseCat([ExtractionState([i,j], true, s.collected, s.pos, s.sample_two), ExtractionState([i,j], false, s.collected, s.sample_one, s.sample_two)], [0.9,0.1]) # 90% chance of successfully scooping
        elseif s.sample_two == [1,1] # not yet collected
            return SparseCat([ExtractionState([i,j], true, s.collected, s.sample_one, s.pos), ExtractionState([i,j], false, s.collected, s.sample_one, s.sample_two)], [0.9,0.1])
        else # third sample
            return SparseCat([ExtractionState([i,j], true, s.collected, s.sample_one, s.sample_two), ExtractionState([i,j], false, s.collected, s.sample_one, s.sample_two)], [0.9,0.1])
        end
    end
    if a == 1 && s.full # accept
        return Deterministic(ExtractionState(s.pos, false, s.collected + 1, s.sample_one, s.sample_two))
    end
    if a == 2 && s.full # reject
        return Deterministic(ExtractionState(s.pos, false, s.collected, s.sample_one, s.sample_two))
    end
    return Deterministic(s) # default (do nothing)
end

function get_grid_cell(n, theta, l)
    # Convert angle from degrees to radians
    theta_rad = radians(theta)
    
    # Calculate Cartesian coordinates
    delta_x = l * sin(theta_rad)
    delta_y = l * cos(theta_rad)
    
    # Calculate grid indices
    i = n - round(delta_y)  # Row index from the bottom
    j = n + round(delta_x)  # Column index from the middle
    
    return [i,j]
end