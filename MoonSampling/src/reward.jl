# Reward
function POMDPs.reward(mdp::ExtractionMDP, s::ExtractionState, a::Int)
    if isterminal(mdp, s)
        return 0.0
    end
    if a in 3:mdp.map_size[1]*mdp.map_size[2]+2 && !s.full
        pos_idx = a - 2
        i = (pos_idx-1) ÷ mdp.map_size[2] + 1
        j = (pos_idx-1) % mdp.map_size[2] + 1
        new_pos = [i,j]
        return -5*norm(new_pos - s.pos)
    end
    if a == 1 && s.full # accept
        if s.pos != s.sample_one && s.pos != s.sample_two
            return mdp.truth_map[s.pos[1], s.pos[2]] ? 500.0 : -1000.0
        else
            return mdp.truth_map[s.pos[1], s.pos[2]] ? 100.0 : -1000.0 # less reward for revisited grid cell
        end
    end
    if a == 2 && s.full # reject
        return mdp.truth_map[s.pos[1], s.pos[2]] ? -10.0 : -1.0
    end
    return -1000000.0 # penalty for invalid action
end