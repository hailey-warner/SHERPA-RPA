using Match

function POMDPs.reward(pomdp::SamplePOMDP, a::Int)
    @match a begin
        # movement
        BASIC_ACTIONS_DICT[:left]   => r += pomdp.step_penalty
        BASIC_ACTIONS_DICT[:right]  => r += pomdp.step_penalty
        BASIC_ACTIONS_DICT[:up]     => r += pomdp.step_penalty
        BASIC_ACTIONS_DICT[:down]   => r += pomdp.step_penalty
        # sampling
        BASIC_ACTIONS_DICT[:scoop]  => r += pomdp.scoop_penalty
        BASIC_ACTIONS_DICT[:accept] => r += is_the_rock_good ? pomdp.accept_good_reward : pomdp.accept_bad_penalty
        BASIC_ACTIONS_DICT[:reject] => r += is_the_rock_good ? pomdp.reject_good_reward : pomdp.reject_bad_penalty
    end
    return r
end