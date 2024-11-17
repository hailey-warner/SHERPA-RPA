const OBSERVATION_NAME = (:good, :bad, :none)

POMDPs.observations(pomdp::SamplePOMDP) = 1:3 # good, bad, none
POMDPs.obsindex(pomdp::SamplePOMDP, o::Int) = o

function POMDPs.observation(pomdp::SamplePOMDP, a::Int, s::State)
    # TODO: update belief map estimate (99% proba. of uncovering truth)
    pomdp.belief_map[s.pos...] = rand() < 0.99 ? pomdp.true_map[s.pos...] : !pomdp.true_map[s.pos...]
    if (a != 5) # not scooping
        return SparseCat((1,2,3),(0.0,0.0,1.0)) # good = 0, bad = 0, none = 1
    else
        # !!!
    end
end