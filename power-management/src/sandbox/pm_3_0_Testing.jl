using POMDPs
using SARSOP
using POMDPModels
using Plots

function plot_test(pomdp,policy,priority_arr_test,inst_battery_usage_test,num_inst)
    # Initialize data and plot
    rewards = Float64[]  # x-coordinates
    counts = Float64[]  # y-coordinates
    states = Float64[]
    battery_levels = Float64[]
    max_idles = Float64[]
    actions = Int[]
    counts_a = Int[]


    up =updater(policy)
    b0 =initialstate(pomdp)
    s = state_toindex(Int(100), 0, (2^num_inst), 100) #rand(initialstate(pomdp))
    b = initialize_belief(up, b0)

    r_total = 0.0
    d = 1.0
    for i in 0:100
        a = action(policy, b)
        
        s, o, r = @gen(:sp,:o,:r)(pomdp, s, a)
        r_total += r
        d *= discount(pomdp)
        println("  state:", s," obs:", o," reward:", r, " r_total: ",r_total," action:", a, " idle_times: ", pomdp.idle_times)
        # print(o)
        b = Deterministic(s) #update(up, b, a, o)
        battery_level, max_idle = state_decompose(10, s)
        push!(counts, i)
        for (index,bool) in pairs(to_fixed_binary(a, num_inst))
            if bool == 1
                push!(counts_a, i)
                push!(actions, index)
            end
        end
        push!(rewards, r_total) 
        push!(states, s)
        push!(battery_levels, battery_level)
        push!(max_idles, max_idle)
    end
    inst_total = 0
    for inst in actions
        inst_total += priority_arr_test[inst]
    end
    println("inst usage: ",inst_total)
    println("reward: ", r_total)

    plot(
        plot(counts, rewards, label="rewards", linewidth=2,title="Rewards over 100 timesteps", xlabel="Timestep", ylabel="Reward"),
        plot(counts, battery_levels, label="battery level", linewidth=2,color=:green,title="Battery Level over 100 timesteps", xlabel="Timestep", ylabel="Battery",ylims=(0,100)),
        scatter(counts_a, actions, label="turned on", markersize=0.5,title="Instruments On", xlabel="Timestep", ylabel="Instrument ID",ylims=(0,6),color=:black),
        # annotate!(5, -0.2, "Subtitle at the Bottom\nWith an Enter", :center, fontsize=10),

        # plot(counts, max_idles, label="maximum idle", linewidth=2,color=:green,title="Maximum Idle Time over 100 timesteps", xlabel="Timestep", ylabel="Maximum Idle Time"),
        
        layout = (1, 3),  # Arrange subplots in 1 row and 2 columns
        size=(1200, 400),   # Set figure size
        suptitle=string("Priority Array: ", join(priority_arr_test, ", ")," \n Instrument Battery Usage: ", join(inst_battery_usage_test, ", "))
        )
        # ,priority_arr_test,inst_battery_usage_test
    
end


function plot_test_v(pomdp,policy_value_iteration)

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
        println("  state:", s," obs:", o," reward:", r, " r_total: ",r_total," action:", a, " idle_times: ", pomdp.idle_times)
        # b = update(up, b, a, o)

        battery_level, max_idle = state_decompose(10, s)
        push!(counts, i)
        push!(rewards, r_total) 
        push!(states, battery_level)
    end

    plot(
        plot(counts, rewards, label="rewards", linewidth=2,title="Rewards over 100 timesteps", xlabel="Timestep", ylabel="Reward"),
        plot(counts, states, label="battery level", linewidth=2,color=:green,title="Battery Level over 100 timesteps", xlabel="Timestep", ylabel="Battery"),
        layout = (1, 2),  # Arrange subplots in 1 row and 2 columns
        size=(1200, 500)   # Set figure size
    )
    
end
#####################################################################


include("../src/power_management3_0.jl")
    # priority_arr = [1,10,10,1,1] #rand(1:10, num_inst)
    # inst_battery_usage = [30, 30, 30, 30, 30]#rand(1:battery_cp, num_inst)
battery_cp = 100
num_inst = 4
priority_arr_test = [5,5,5,5]#rand(1:10, num_inst)#[1,2,3,4,5]
inst_battery_usage_test = [30, 30, 30, 30]#rand(1:battery_cp, num_inst)#[10, 20, 30, 40, 50]

pomdp = PowerManagementPOMDP(num_inst = num_inst, 
                            idle_time_max = 10, 
                            charging = 100,  
                            battery_cp = battery_cp,
                            priority_arr = priority_arr_test,
                            inst_battery_usage = inst_battery_usage_test)
solver = SARSOPSolver(verbose = true, timeout=100)
policy = solve(solver, pomdp)
plot_test(pomdp,policy,priority_arr_test,inst_battery_usage_test,num_inst)


using DiscreteValueIteration
solver = ValueIterationSolver(max_iterations=100, belres=1e-6, verbose=true) # creates the solver
mdp = UnderlyingMDP(pomdp)
policy_value_iteration = solve(solver, mdp) # runs value iterations


using MCTS
# using StaticArrays
solver = MCTSSolver(n_iterations=10000, depth=100, exploration_constant=10.0)
mdp = UnderlyingMDP(pomdp)
planner = solve(solver, mdp)
s = 100#rand(initialstate(pomdp))
a = action(planner, s) # returns the action for state s

#####################################################################


policy = load_policy(pomdp,"policy.out")
simulator = SARSOPSimulator(sim_num = 5, sim_len = 5, 
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

#####################################################################

plot_test(pomdp,policy,priority_arr_test,inst_battery_usage_test,num_inst)
plot_test_v(pomdp,policy_value_iteration)
histogram(policy.action_map, bins=30, xlabel="Value", ylabel="Frequency", title="Histogram Example")
