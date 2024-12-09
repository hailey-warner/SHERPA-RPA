using POMDPs
using SARSOP
using POMDPModels


include("../src/power_management2_0.jl")
# include("../src/power_management.jl")

pomdp = PowerManagementPOMDP(num_inst = 5)
solver = SARSOPSolver(verbose = true, timeout=100)
policy = solve(solver, pomdp)

#####################################################################

# if I want to use value iteration
using DiscreteValueIteration
solver = ValueIterationSolver(max_iterations=100, belres=1e-6, verbose=true) # creates the solver
mdp = UnderlyingMDP(pomdp)
policy_value_iteration = solve(solver, mdp) # runs value iterations

#####################################################################


pomdp = PowerManagementPOMDP(num_inst = 5)
policy = load_policy(pomdp,"policy.out")

simulator = SARSOPSimulator(sim_num = 5, sim_len = 100, 
                            policy_filename = "policy.out",
                            pomdp_filename = "model.pomdpx")
simulate(simulator) 
# evaluate the SARSOP policy
evaluator = SARSOPEvaluator(sim_num = 5, sim_len = 10, 
                            policy_filename = "policy.out",
                            pomdp_filename = "model.pomdpx")
# evaluate(evaluator)

# generates a policy graph
graphgen = PolicyGraphGenerator(graph_filename = "SARSOP_battery.dot",
                                policy_filename = "policy.out",
                                pomdp_filename = "model.pomdpx")
generate_graph(graphgen)


using Plots

# Initialize data and plot
rewards = Float64[]  # x-coordinates
counts = Float64[]  # y-coordinates
states = Float64[]
battery_levels = Float64[]
max_idles = Float64[]

# b0 =initialstate(pomdp)
s = rand(initialstate(pomdp))

r_total = 0.0
d = 1.0
for i in 0:100
    a = action(policy_value_iteration, s)
    s, o, r = @gen(:sp,:o,:r)(pomdp, s, a)
    r_total += r
    d *= discount(pomdp)
    println("  state:", s," obs:", o," reward:", r, " r_total: ",r_total," action:", a)
    # b = update(up, b, a, o)
    push!(counts, i)
    push!(rewards, r_total) 
    push!(states, s)
end

plot(
    plot(counts, rewards, label="rewards", linewidth=2,title="Rewards over 100 timesteps", xlabel="Timestep", ylabel="Reward"),
    plot(counts, states, label="battery level", linewidth=2,color=:green,title="Battery Level over 100 timesteps", xlabel="Timestep", ylabel="Battery"),
    layout = (1, 2),  # Arrange subplots in 1 row and 2 columns
    size=(1200, 400)   # Set figure size
)


#####################################################################

# Initialize data and plot
rewards = Float64[]  # x-coordinates
counts = Float64[]  # y-coordinates
states = Float64[]
battery_levels = Float64[]
max_idles = Float64[]


up =updater(policy)
b0 =initialstate(pomdp)
s = rand(initialstate(pomdp))
b = initialize_belief(up, b0)

r_total = 0.0
d = 1.0
for i in 0:100
    a = action(policy, b)
    s, o, r = @gen(:sp,:o,:r)(pomdp, s, a)
    r_total += r
    d *= discount(pomdp)
    println("  state:", s," obs:", o," reward:", r, " r_total: ",r_total," action:", a)
    b = update(up, b, a, o)
    push!(counts, i)
    push!(rewards, r_total) 
    push!(states, s)
end

plot(
    plot(counts, rewards, label="rewards", linewidth=2,title="Rewards over 100 timesteps", xlabel="Timestep", ylabel="Reward"),
    plot(counts, states, label="battery level", linewidth=2,color=:green,title="Battery Level over 100 timesteps", xlabel="Timestep", ylabel="Battery"),
    layout = (1, 2),  # Arrange subplots in 1 row and 2 columns
    size=(1200, 400)   # Set figure size
)

