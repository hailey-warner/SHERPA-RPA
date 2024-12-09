https://github.com/sisl/MultiAgentPOMDPProblems.jl/blob/main/src/multi_tiger_pomdp.jl
>> very helpful from dylan examples of his multi_pomdp one

https://juliapomdp.github.io/POMDPs.jl/stable/def_pomdp/
>> basic how to build out pomdp

https://github.com/JuliaPOMDP/SARSOP.jl
>> sarsop.jl github

https://github.com/JuliaPOMDP/POMDPModels.jl/blob/master/src/CryingBabies.jl
>> pomdpmodels.jl

https://juliapomdp.github.io/POMDPs.jl/latest/POMDPTools/testing/
>> pomdptools for testing

opt j o for repl
opt j k for kill repl

https://github.com/JuliaPOMDP/PointBasedValueIteration.jl
- value iteration for pomdp



# notes 11/20/24

Trying to check a few things:
- keeping poewr generation as a constant (failure cases will be changed to make power generation not be optimal all of the time)
- fixed the reward function, I think it got tofix amny of the bugs that were originally there (I think possibly something about the way I was computing idle time and whatnot)
- what is the continous space method we want to use and compare? (MCVI)?? 
    - https://github.com/JuliaPOMDP/MCVI.jl
- explicit method and this and sarsop??

Things I can plot now with this information:
- maybe we could plot computation
- but more important is the performance
    - different simulation results
    - what are the metrics we care about?




- need to change the belief (to a sparsecat so that we can now keep beliefs to 0-1)
    - Observation / Transition ( distributions should also be 1)
    - use dylan's code, the truncated normal (on his multiagent pomdps.jl (joint meat problem?))



little commands:
dot -Tpng SARSOP_battery.dot > output.png

# notes 12/5/24

(COMPLETED)
- dot thing works, should only try with deterministic observations though
- sparsecat distributions i think also works now for a perturbation around the belief part


(NEED TO DO)
- random seed for constant testing ( check if it actually is working?)
- 

# notes 12/5/24

- sarsop can't take invalid actions how can we set this up
- also tryint to set up reward funciton so it doesn't take invalid actions but also tries to choose good idle things

- sarsop but recomputed every few iterations? is that possible?


(NEED TO DO)
- make sure to make pomdp that actually is with observable function
    - and try it with sarsop
    

