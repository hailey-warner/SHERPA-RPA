using POMDPs
using SARSOP
using POMDPModels


include("../src/power_management2_0.jl")
# include("../src/power_management.jl")

pomdp = PowerManagementPOMDP(num_inst = 5)
# pomdp = PowerManagementPOMDP(num_inst = 5, idle_time_max = 10, charging = 50,  battery_cp = 100)
solver = SARSOPSolver(verbose = true, timeout=100)

policy = solve(solver, pomdp)


using DiscreteValueIteration
solver = ValueIterationSolver(max_iterations=100, belres=1e-6, verbose=true) # creates the solver

mdp = UnderlyingMDP(pomdp)
policy = solve(solver, mdp) # runs value iterations


using POMDPLinter

@show_requirements POMDPs.solve(solver, pomdp)
# @show_requirements POMDPs.solve(solver, mdp)
using POMDPTools

POMDPTools.Testing.has_consistent_initial_distribution(pomdp)


# # using Revise

# # include("../src/power_management2_0.jl")
# include("../src/power_management.jl")

# # include("../src/inverted_pendulum.jl")
# # using PowerManagementPOMDP

# # include("./example.jl")
# # using .example: TigerPOMDP
# using POMDPs
# using SARSOP

# using POMDPModels
# # using NativeSARSOP
# # using POMDPModels
# # mdp = InvertedPendulum()
# # mdp = SimpleGridWorld()
# pomdp = PowerManagementPOMDP(num_inst = 5)
# # pomdp_tiger = TigerPOMDP(0.85)
# println(pomdp)

# # do what you would do with a POMDP model, for example use QMDP to solve it
# # using QMDP
# # solver = QMDPSolver(verbose=true) 
# solver = SARSOPSolver(verbose = true)

# # policy = solve(solver, pomdp_tiger)
# policy = solve(solver, pomdp)
# # print(policy)

# using DiscreteValueIteration
# solver = ValueIterationSolver(max_iterations=100, belres=1e-6, verbose=true) # creates the solver
# policy = solve(solver, mdp)

# # TigerPOMDP(p_correct=0.85) = new(p_correct, Dict("left"=>1, "right"=>2, "listen"=>3))
# using POMDPLinter

# @show_requirements POMDPs.solve(solver, pomdp)
# @show_requirements POMDPs.solve(solver, mdp)
# using POMDPTools

# POMDPTools.Testing.has_consistent_initial_distribution(pomdp)