using POMDPs
using SARSOP
using POMDPModels
using DiscreteValueIteration
using Plots
include("../src/models/power_management_clean.jl")
include("../src/plotting.jl")
include("../src/policies/deterministic.jl")


# POMDP Initialization 
battery_cp = 100
num_inst = 4
priority_arr_test = [1,1,1,1] #rand(1:10, num_inst)#[1,2,3,4,5]
inst_battery_usage_test = [30, 30, 30, 30] #rand(1:battery_cp, num_inst)#[10, 20, 30, 40, 50]

pomdp = PowerManagementPOMDP(num_inst = num_inst, 
                            idle_time_max = 10, 
                            charging = 100,  
                            battery_cp = battery_cp,
                            priority_arr = priority_arr_test,
                            inst_battery_usage = inst_battery_usage_test)

#####################################################################

# SARSOP Specific Solver
solver = SARSOPSolver(verbose = true, timeout=100)
policy = solve(solver, pomdp)
plot_SARSOP_test(pomdp, policy, 100, true)

#####################################################################

# Value Iteration Specific Solver
solver = ValueIterationSolver(max_iterations=100, belres=1e-6, verbose=true) # creates the solver
mdp = UnderlyingMDP(pomdp)
policy_value_iteration = solve(solver, mdp) # runs value iterations
plot_VI_test(pomdp,policy_value_iteration,100,true)

#####################################################################

# Deterministic
plot_deterministic_test(pomdp,deterministic_policy,100,true)
