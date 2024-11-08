

# using Revise


include("../src/power_management.jl")
# using PowerManagementPOMDP

# include("./example.jl")
# using .example: TigerPOMDP
using POMDPs
using SARSOP
# using NativeSARSOP
# using POMDPModels

pomdp = PowerManagementPOMDP(num_inst = 3)
# pomdp_tiger = TigerPOMDP(0.85)
println(pomdp)

# do what you would do with a POMDP model, for example use QMDP to solve it
# using QMDP
# solver = QMDPSolver(verbose=true) 
solver = SARSOPSolver()

# policy = solve(solver, pomdp_tiger)
policy = solve(solver, pomdp)

# print(policy)

# TigerPOMDP(p_correct=0.85) = new(p_correct, Dict("left"=>1, "right"=>2, "listen"=>3))
using POMDPLinter

@show_requirements POMDPs.solve(solver, pomdp)

using POMDPTools

POMDPTools.Testing.has_consistent_initial_distribution(pomdp)