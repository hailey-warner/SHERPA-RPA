function POMDPs.reward(m::PowerManagementPOMDP, s, a)

    reward = 0.0


    battery_level, max_idle = state_decompose(m.idle_time_max, s)

    # negative rewards for idle times
    action_vec = reverse(to_fixed_binary(a, m.num_inst))
    flipped_action = map(x -> 1 - x, action_vec)
    reward -= dot(m.priority_arr,flipped_action)* max_idle
    
    # reward -= max_idle*m.idle_time_max_penalty

    # reward -= max_idle*40#dot(m.priority_arr,transpose(m.idle_times))*100

    if sum(m.inst_battery_usage) > m.battery_cp && sum(action_vec) == m.num_inst
        reward -= m.exceeded_capacity
    else
        # negative rewards for battery used past capacity assigning too many instruments
        if battery_level == 0
            reward -= m.exceeded_capacity
        end
    end


    return reward
end

################################(FAILEEDDDDD)############################

# julia plot 1
function POMDPs.reward(m::PowerManagementPOMDP, s, a)

    reward = 0.0

    battery_level, max_idle = state_decompose(m.idle_time_max, s)

    if battery_level == 0
        reward -= m.exceeded_capacity
    end
    reward -= sum(m.idle_times)

    return reward
end

# julia plot 2,3

function POMDPs.reward(m::PowerManagementPOMDP, s, a)

    reward = 0.0

    battery_level, max_idle = state_decompose(m.idle_time_max, s)

    if battery_level == 0
        reward -= m.exceeded_capacity
    else
        # negative rewards for idle times
        action_vec = reverse(to_fixed_binary(a, m.num_inst))
        flipped_action = map(x -> 1 - x, action_vec)
        reward -= dot(m.priority_arr,flipped_action)* max_idle
    end


    return reward
end

# julia plot 4,5
#PowerManagementPOMDP(5, [7, 5, 5, 8, 7], [17, 62, 67, 46, 30], 100, 5, 50, 10, [0, 101, 102, 11, 0], 100.0, 0.9, 10.0)

function POMDPs.reward(m::PowerManagementPOMDP, s, a)

    reward = 0.0

    battery_level, max_idle = state_decompose(m.idle_time_max, s)

    if battery_level == 0
        reward -= m.exceeded_capacity
    else
        # negative rewards for idle times
        action_vec = reverse(to_fixed_binary(a, m.num_inst))
        flipped_action = map(x -> 1 - x, action_vec)
        reward -= dot(m.priority_arr,flipped_action) #* max_idle
    end


    return reward
end

# julia plot 6,7
# PowerManagementPOMDP(5, [7, 5, 5, 8, 7], [17, 62, 67, 46, 30], 100, 5, 50, 10, [0, 101, 102, 11, 0], 100.0, 0.9, 10.0)

function POMDPs.reward(m::PowerManagementPOMDP, s, a)

    reward = 0.0

    battery_level, max_idle = state_decompose(m.idle_time_max, s)

    if battery_level == 0
        reward -= m.exceeded_capacity
    else
        # negative rewards for idle times
        action_vec = reverse(to_fixed_binary(a, m.num_inst))
        flipped_action = map(x -> 1 - x, action_vec)
        reward -= dot(m.priority_arr,flipped_action)
    end

    if count(x -> x > max_idle, m.idle_times) > 1
        reward -= max_idle* count(x -> x > max_idle, m.idle_times)
    end


    return reward
end