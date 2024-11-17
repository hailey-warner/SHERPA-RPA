using Match

function POMDPs.reward(pomdp::SamplePOMDP, s::State, a::Int)
    r = @match a begin
        # movement
        1 => pomdp.step_penalty
        2 => pomdp.step_penalty
        3  => pomdp.step_penalty
        4  => pomdp.step_penalty
        # sampling
        5 => pomdp.scoop_penalty
        6 => pomdp.true_map[s.pos...] == 1 ? pomdp.accept_good_reward : pomdp.accept_bad_penalty
        7 => pomdp.true_map[s.pos...] == 1 ? pomdp.reject_good_reward : pomdp.reject_bad_penalty
    end
    return r
end