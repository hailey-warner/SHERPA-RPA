import Pkg
Pkg.add("~/hlwarner/SISL/SHERPA-RPA/MoonSampling")

using POMDPs
using POMDPTools
using StaticArrays
using Distributions
using LinearAlgebra
using GaussianProcesses
using Random
using MCTS
using DiscreteValueIteration
using SARSOP
using Plots
using MoonSampling # user-defined package
Random.seed!(1)

# Example usage
function simulate_policy(mdp, policy, truth_map; max_steps=30)
    s = ExtractionState([1, 1], false, 0, [1,1], [1,1])  # initial state
    reward = 0.0
    steps = 0
    total_reward = Float64[]
    
    while !isterminal(mdp, s) && steps < max_steps
        steps += 1
        a = action(policy, s)
        sp = rand(transition(mdp, s, a))
        r = reward(mdp, s, a)
        println("State: pos=$(s.pos), full=$(s.full), collected=$(s.collected) → Action: $a → Reward: $r")
        reward += r
        s = sp
        push!(total_reward, reward)
        push!(maps_over_time, truth_map)
        push!(pos_over_time, s.pos)
        push!(action_over_time, a)
    end
    if steps >= max_steps
        println("Simulation stopped after reaching max_steps.")
    end
    #println("Final state: pos=$(s.pos), full=$(s.full), collected=$(s.collected)")
    println("Reward: $reward")``
    push!(reward_over_time, reward)

    plot!(total_reward, linewidth=2, alpha=0.7, xlims=(0,10))
end

function create_gif(maps_over_time, pos_over_time, action_over_time; filename="animation.gif", fps=2)
    anim = @animate for (i, frame) in enumerate(maps_over_time)
        heatmap(frame, color=:blues, legend=false, yflip=true, axis=false, ticks=false, aspect_ratio=:equal)
        color = :grey
        if action_over_time[i] == 1
            color = :green
        elseif action_over_time[i] == 2
            color = :red
        end
        scatter!([pos_over_time[i][2]],[pos_over_time[i][1]],markersize=10,markershape=:circle,markercolor=color,legend=false, yflip=true)
    end
    gif(anim, filename, fps=fps)
 end


# initalizing simulation 
maps_over_time = []
pos_over_time = []
action_over_time = []
reward_over_time = []
runtimes = []

p = plot(xlabel="Step",
         ylabel="Cumulative Reward",
         legend=false)

# generating random truth maps
truth_maps = [rand(Bool, 10, 10) for _ in 1:10]

# run solver for all truth maps
for truth_map in truth_maps
    time = @elapsed begin
    mdp = ExtractionMDP(SA[10,10], truth_map)
    solver = MCTSSolver(n_iterations=1000, depth=5, exploration_constant=1.0)
    #solver = ValueIterationSolver(max_iterations=1000)
    #solver = RandomSolver()
    policy = solve(solver, mdp)
    simulate_policy(mdp, policy, truth_map)
    end
    push!(runtimes, time)
end
savefig("total_reward.png")

create_gif(maps_over_time, pos_over_time, action_over_time)