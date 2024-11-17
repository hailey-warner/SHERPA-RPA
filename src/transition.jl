using Match

function POMDPs.transition(pomdp::SamplePOMDP, s::State, a::Int)
    if isterminal(pomdp, s)
        return Deterministic(pomdp.terminal_state)
    end
    # movement
    new_pos = @match a begin # TODO: add bounds
        1 => (-1,0)
        2 => (1, 0)
        3 => (0,-1)
        4 => (0, 1)
        _ => (0, 0)
    end
    new_pos += s.pos
    # sampling
    new_collected = s.collected
    new_full = s.full
    if a == BASIC_ACTIONS_DICT[:scoop]
        new_full = true
    end     
    if a == BASIC_ACTIONS_DICT[:accept]
        new_full = false
        new_collected += 1
    end
    if a == BASIC_ACTIONS_DICT[:reject]
        new_full = false
    end
    new_state = State(new_pos, new_full, new_collected)
    return Deterministic(new_state)
end