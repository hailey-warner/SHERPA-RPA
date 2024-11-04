function POMDPs.transition(pomdp::SamplePOMDP, s::State, a::Int)
    @match a begin
        # movement
        BASIC_ACTIONS_DICT[:left]   => s.pos += (-1,0)
        BASIC_ACTIONS_DICT[:right]  => s.pos += (1, 0)
        BASIC_ACTIONS_DICT[:up]     => s.pos += (0,-1)
        BASIC_ACTIONS_DICT[:down]   => s.pos += (0, 1)

        # sampling
        BASIC_ACTIONS_DICT[:scoop] => begin
            state.occupied = true
            # TODO: update quality and confidence map estimates
            s.qual_map = None
            s.conf_map = None      
        end     

        BASIC_ACTIONS_DICT[:accept] => begin
            s.occupied = 0
            s.collected += 1
        end

        BASIC_ACTIONS_DICT[:reject] => s.occupied = 0
    end
end