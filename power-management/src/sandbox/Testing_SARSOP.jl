
using POMDPs
using SARSOP

include("../src/power_management2_0.jl")
# include("../src/power_management3_0.jl")

pomdp = PowerManagementPOMDP(num_inst = 5)

# pomdp = PowerManagementPOMDP(num_inst = 5, idle_time_max = 10, charging = 50,  battery_cp = 100)
println(pomdp)

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



# b = uniform_belief(pomdp) # from POMDPModelTools
# a = action(policy, b) 
# b = 


# up =updater(policy)
# b0 =initialstate(pomdp)
# s = rand(initialstate(pomdp))
# b = initialize_belief(up, b0)

# r_total = 0.0
# d = 1.0
# for i in 0:5
#     a = action(policy, b)
#     s, o, r = @gen(:sp,:o,:r)(pomdp, s, a)
#     r_total += d*r
#     d *= discount(pomdp)
#     println("  state:", s," obs:", o," reward:", r, " r_total: ",r_total," action:", a)
#     b = update(up, b, a, o)
# end

# r_total
# function to_fixed_binary(n::Int, num_digits::Int)
#     binary_digits = digits(n, base =2)  # Convert to binary
#     padded_binary =  vcat(zeros(Int, num_digits - length(binary_digits)), binary_digits)  # Pad with leading zeros if needed
#     return padded_binary
# end

#####################################################################

using Plots

# pomdp = PowerManagementPOMDP(num_inst = 5, idle_time_max = 10, charging = 50,  battery_cp = 100)

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
    println("  state:", s," obs:", o," reward:", r, " r_total: ",r_total," action:", a)#, " idle_times: ", pomdp.idle_times)
    # print(o)
    b = update(up, b, a, o)
    # battery_level, max_idle = state_decompose(10, s)
    push!(counts, i)
    push!(rewards, r_total) 
    push!(states, s)
    # push!(battery_levels, battery_level)
    # push!(max_idles, max_idle)
end


plot(
    plot(counts, rewards, label="rewards", linewidth=2,title="Rewards over 100 timesteps", xlabel="Timestep", ylabel="Reward"),
    plot(counts, states, label="battery level", linewidth=2,color=:green,title="Battery Level over 100 timesteps", xlabel="Timestep", ylabel="Battery"),
    # plot(counts, max_idles, label="maximum idle", linewidth=2,color=:green,title="Battery Level over 100 timesteps", xlabel="Timestep", ylabel="Maximum Idle Time"),
    layout = (1, 3),  # Arrange subplots in 1 row and 2 columns
    size=(1200, 400)   # Set figure size
)
# plt = plot(counts,rewards, xlabel="Iteration", ylabel="Value", legend=false, title="Dynamic Plot")
#####################################################################

s = rand(initialstate(pomdp))

r_total = 0.0
d = 1.0
for i in 0:100
    a = action(policy, s)
    s, o, r = @gen(:sp,:o,:r)(pomdp, s, a)
    println(reward(pomdp, s, a))
    println(pomdp.idle_times)
    r_total += r
    d *= discount(pomdp)
    println("  state:", s," obs:", o," reward:", r, " r_total: ",r_total," action:", a)
    # print(o)
    b = Deterministic(o) #update(up, b, a, o)
    battery_level, max_idle = state_decompose(10, s)
    push!(counts, i)
    push!(rewards, r_total) 
    push!(states, s)
    push!(battery_levels, battery_level)
    push!(max_idles, max_idle)
end


plot(
    plot(counts, rewards, label="rewards", linewidth=2,title="Rewards over 100 timesteps", xlabel="Timestep", ylabel="Reward"),
    plot(counts, battery_levels, label="battery level", linewidth=2,color=:green,title="Battery Level over 100 timesteps", xlabel="Timestep", ylabel="Battery"),
    plot(counts, max_idles, label="maximum idle", linewidth=2,color=:green,title="Battery Level over 100 timesteps", xlabel="Timestep", ylabel="Maximum Idle Time"),
    layout = (1, 3),  # Arrange subplots in 1 row and 2 columns
    size=(1200, 400)   # Set figure size
)

# to_fixed_binary(21, pomdp.num_inst)

# dot(pomdp.priority_arr,transpose(action_vec))

# sp = s+rand(Uniform(0: pomdp.battery_cp))
# if sp > 100
#     sp = 100
# end
# action_vec
# for (a, ind) in zip(action_vec, 1:length(action_vec))
#     sp -= a*pomdp.inst_battery_usage[ind] #rand(Normal(m.inst_battery_usage[ind], m.inst_battery_variance[ind])) # is it bad to put in randomness in how much we are using battery?
#     if sp < 0
#         sp = 0
#         break
#     end
# end
# sp