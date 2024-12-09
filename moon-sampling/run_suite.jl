using POMDPs
using POMDPTools
using StaticArrays
using Distributions
using MCTS
using LinearAlgebra
using Plots
using DiscreteValueIteration
using SARSOP
using GaussianProcesses
using Random
Random.seed!(84)

struct ExtractionState
    pos::Vector{Int}
    full::Bool
    collected::Int
    sample_one::Vector{Int}
    sample_two::Vector{Int}
end

mutable struct ExtractionMDP <: MDP{ExtractionState, Int}
    map_size::SVector{2, Int}
    truth_map::Array{Bool, 2}
end

# States
function Base.length(mdp::ExtractionMDP)
    return (mdp.map_size[1] * mdp.map_size[2])^3 * 2 * 4
end

POMDPs.states(mdp::ExtractionMDP) = [
    ExtractionState([i,j], full, collected, [a,b], [c,d])
    for i in 1:mdp.map_size[1]
    for j in 1:mdp.map_size[2]
    for full in [false, true]
    for collected in 0:3
    for a in 1:mdp.map_size[1]
    for b in 1:mdp.map_size[2] 
    for c in 1:mdp.map_size[1]
    for d in 1:mdp.map_size[2]
]

POMDPs.initialstate(mdp::ExtractionMDP) = Deterministic(ExtractionState([1, 1], false, 0, [0,0], [1,1]))

function POMDPs.convert_s(T::Type{<:AbstractArray}, s::ExtractionState, mdp::ExtractionMDP)
    return convert(T, vcat(s.pos, s.full, s.collected, s.sample_one, s.sample_two))
end

function POMDPs.convert_s(T::Type{ExtractionState}, v::AbstractArray, mdp::ExtractionMDP)
    return ExtractionState([v[1], v[2]], Bool(v[3]), Int(v[4]), [v[5], v[6]], [v[7], v[8]])
end

function POMDPs.isterminal(mdp::ExtractionMDP, s::ExtractionState)
    return s.collected >= 3  # Terminate when we have 3 samples onboard
end

function POMDPs.stateindex(mdp::ExtractionMDP, s::ExtractionState)
    i, j = s.pos
    a, b = s.sample_one
    c, d = s.sample_two
    n_cols = mdp.map_size[2]
    
    # calculate base indices for each component
    pos_idx = (i-1) * n_cols + (j-1)
    full_idx = s.full ? 1 : 0
    collected_idx = s.collected
    sample_one_idx = (a-1) * n_cols + (b-1)
    sample_two_idx = (c-1) * n_cols + (d-1)
    
    return 1 + pos_idx + 
           full_idx * (mdp.map_size[1] * mdp.map_size[2]) + 
           collected_idx * (mdp.map_size[1] * mdp.map_size[2] * 2) +
           sample_one_idx * (mdp.map_size[1] * mdp.map_size[2] * 2 * 4) +
           sample_two_idx * (mdp.map_size[1] * mdp.map_size[2] * 2 * 4 * (mdp.map_size[1] * mdp.map_size[2]))
end

# Actions
# [ accept, reject, scoop11, scoop12, ..., scoopmn ]
POMDPs.actions(mdp::ExtractionMDP) = collect(1:(mdp.map_size[1]*mdp.map_size[2]+2))

function POMDPs.actionindex(mdp::ExtractionMDP, a::Int)
    return a
end

# Transitions
function POMDPs.transition(mdp::ExtractionMDP, s::ExtractionState, a::Int)
    if isterminal(mdp, s)
        return Deterministic(s)
    end
    if a in 3:mdp.map_size[1]*mdp.map_size[2]+2 && !s.full
        pos_idx = a - 2
        i = (pos_idx-1) ÷ mdp.map_size[2] + 1
        j = (pos_idx-1) % mdp.map_size[2] + 1
        if s.sample_one == [1,1] # 90% chance of successfully scooping
            return SparseCat([ExtractionState([i,j], true, s.collected, s.pos, s.sample_two), ExtractionState([i,j], false, s.collected, s.sample_one, s.sample_two)], [0.9,0.1])
        elseif s.sample_two == [1,1]
            return SparseCat([ExtractionState([i,j], true, s.collected, s.sample_one, s.pos), ExtractionState([i,j], false, s.collected, s.sample_one, s.sample_two)], [0.9,0.1])
        else # third sample
            return SparseCat([ExtractionState([i,j], true, s.collected, s.sample_one, s.sample_two), ExtractionState([i,j], false, s.collected, s.sample_one, s.sample_two)], [0.9,0.1])
        end
    end
    if a == 1 && s.full # accept
        return Deterministic(ExtractionState(s.pos, false, s.collected + 1, s.sample_one, s.sample_two))
    end
    if a == 2 && s.full # reject
        return Deterministic(ExtractionState(s.pos, false, s.collected, s.sample_one, s.sample_two))
    end
    return Deterministic(s) # default (do nothing)
end

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

# Discount
POMDPs.discount(mdp::ExtractionMDP) = 0.95

# Example usage
function simulate_policy(mdp, policy, truth_map; max_steps=30)
    s = ExtractionState([1, 1], false, 0, [1,1], [1,1])  # Start state
    total_reward = 0.0
    steps = 0
    cumulative_rewards = Float64[]
    
    while !isterminal(mdp, s) && steps < max_steps
        steps += 1
        a = action(policy, s)
        sp = rand(transition(mdp, s, a))
        r = reward(mdp, s, a)
        println("State: pos=$(s.pos), full=$(s.full), collected=$(s.collected) → Action: $a → Reward: $r")
        total_reward += r
        s = sp
        push!(cumulative_rewards, total_reward)
        push!(maps_over_time, truth_map)
        push!(pos_over_time, s.pos)
        push!(action_over_time, a)
    end
    if steps >= max_steps
        println("Simulation stopped after reaching max_steps.")
    end
    println("Truth map: $truth_map")
    println("Final state: pos=$(s.pos), full=$(s.full), collected=$(s.collected)")
    println("Total reward: $total_reward")
    push!(final_rewards, total_reward)

    plot!(cumulative_rewards, linewidth=2, alpha=0.7, xlims=(0,10))
end

function create_gif(maps_over_time, pos_over_time, action_over_time; filename="animation.gif", fps=2)
    anim = @animate for (i, frame) in enumerate(maps_over_time)
        heatmap(frame, color=:blues, legend=false, yflip=true, axis=true, ticks=true, aspect_ratio=:equal)
        color = :grey
        if action_over_time[i] == 1
            color = :green
        elseif action_over_time[i] == 2
            color = :red
        end
        scatter!([pos_over_time[i][2]],[pos_over_time[i][1]],markersize=15,markershape=:circle,markercolor=color,legend=false, yflip=true)
    end
    gif(anim, filename, fps=fps)
 end


# initalizing simulation 
maps_over_time = []
pos_over_time = []
action_over_time = []
final_rewards = []
runtimes = []

p = plot(xlabel="Step",
         ylabel="Cumulative Reward",
         legend=false)

# generating random truth maps
truth_maps = [rand(Bool, 5, 5) for _ in 1:10]

# run solver for all truth maps
for truth_map in truth_maps
    time = @elapsed begin
    mdp = ExtractionMDP(SA[5,5], truth_map)
    #solver = MCTSSolver(n_iterations=50000, depth=3, exploration_constant=1.0)
    solver = ValueIterationSolver(max_iterations=1000)
    #solver = RandomSolver()
    policy = solve(solver, mdp)
    simulate_policy(mdp, policy, truth_map)
    end
    push!(runtimes, time)
end
savefig("cumulative_rewards.png")

create_gif(maps_over_time, pos_over_time, action_over_time)