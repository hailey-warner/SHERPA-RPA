function deterministic_policy(pomdp,s)
    temp_priority_array = copy(pomdp.priority_arr)
    # if any are idle for too long bring to top of priority list
    for (idle, ind) in zip(pomdp.idle_times, 1:length(pomdp.idle_times))
        if idle > 5 
            temp_priority_array[ind] += 10
        end
    end

    # order in priority level and turn on in that order
    order = sortperm(temp_priority_array, rev=true)

    battery_level, prev_action = state_decompose((2^pomdp.num_inst), s)
    temp = 50#  battery_level

    actions_vec = zeros(Int,pomdp.num_inst)
    for indx in order
        if temp - pomdp.inst_battery_usage[indx] > 0
            actions_vec[indx] = 1
            temp = temp - pomdp.inst_battery_usage[indx]
        end
    end

    # action_flipped = map(x -> 1 - x, actions_vec) 

    # convert to action index:
    action_total = 0
    for (act, i) in zip(actions_vec,0:pomdp.num_inst-1)
        if act == 1
            action_total += 2^i
        end
    end 
    return action_total
end
