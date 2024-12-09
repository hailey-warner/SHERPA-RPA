include("models/power_management_clean.jl")

function plot_SARSOP_test(pomdp,policy,start_state,verbose = true)

    # Initialize data for plotting
    rewards = Float64[]  # x-coordinates
    counts = Float64[]  # y-coordinates
    states = Float64[]
    battery_levels = Float64[]
    actions = Int[]
    counts_a = Int[]

    # Initialization of POMDP state and beliefs 
    up =updater(policy)
    b0 = initialstate(pomdp)
    s = state_toindex(Int(start_state), 0, (2^pomdp.num_inst)) #rand(initialstate(pomdp))
    b = initialize_belief(up, b0)

    # initialization of simulation 
    r_total = 0.0
    d = 1

    # for 100 time steps
    for i in 0:100
        
        # generate an action given initial belief
        a = action(policy, b)
        
        # progress to next state with action
        s, o, r = @gen(:sp,:o,:r)(pomdp, s, a)
        d *= discount(pomdp)
        b = Deterministic(s) #update(up, b, a, o)

        # for debugging purposes
        if verbose
            println("  state:", s," obs:", o," reward:", r, " r_total: ",r_total," action:", a, " idle_times: ", pomdp.idle_times)
        end

        # All for plotting and saving data
        r_total += r
        battery_level, prev_action = state_decompose(2^pomdp.num_inst, s)
        push!(counts, i)
        push!(rewards, r_total) 
        push!(states, s)
        push!(battery_levels, battery_level)
        for (index,bool) in pairs(to_fixed_binary(a, pomdp.num_inst))
            if bool == 1
                push!(counts_a, i)
                push!(actions, index)
            end
        end
    end

    # Final statistics
    inst_total = 0
    for inst in actions
        inst_total += pomdp.priority_arr[inst]
    end
    println("inst usage: ",inst_total)
    println("reward: ", r_total)

    # Plotting 
    plot(
        plot(counts, rewards, label="rewards", linewidth=2,title="Rewards over 100 timesteps", xlabel="Timestep", ylabel="Reward"),
        plot(counts, battery_levels, label="battery level", linewidth=2,color=:green,title="Battery Level over 100 timesteps", xlabel="Timestep", ylabel="Battery",ylims=(0,100)),
        scatter(counts_a, actions, label="turned on", markersize=0.5,title="Instruments On", xlabel="Timestep", ylabel="Instrument ID",ylims=(0,6),color=:black),
        layout = (1, 3),  # Arrange subplots in 1 row and 2 columns
        size=(1200, 400),   # Set figure size
        suptitle=string("SARSOP\n Priority Array: ", join(pomdp.priority_arr, ", ")," \n Instrument Battery Usage: ", join( pomdp.inst_battery_usage, ", "))
        )
    
end

function plot_VI_test(pomdp,policy,start_state,verbose = true)

    # Initialize data for plotting
    rewards = Float64[]  # x-coordinates
    counts = Float64[]  # y-coordinates
    states = Float64[]
    battery_levels = Float64[]
    actions = Int[]
    counts_a = Int[]

    # Initialization of state 
    s = state_toindex(Int(start_state), 0, (2^pomdp.num_inst)) #rand(initialstate(pomdp))

    # initialization of simulation 
    r_total = 0.0
    d = 1

    # for 100 time steps
    for i in 0:100
        
        # generate an action given initial state
        a = action(policy, s)
        
        # progress to next state with action
        s, o, r = @gen(:sp,:o,:r)(pomdp, s, a)
        d *= discount(pomdp)

        # for debugging purposes
        if verbose
            println("  state:", s," obs:", o," reward:", r, " r_total: ",r_total," action:", a, " idle_times: ", pomdp.idle_times)
        end

        # All for plotting and saving data
        r_total += r
        battery_level, prev_action = state_decompose(2^pomdp.num_inst, s)
        push!(counts, i)
        push!(rewards, r_total) 
        push!(states, s)
        push!(battery_levels, battery_level)
        for (index,bool) in pairs(to_fixed_binary(a, pomdp.num_inst))
            if bool == 1
                push!(counts_a, i)
                push!(actions, index)
            end
        end
    end

    # Final statistics
    inst_total = 0
    for inst in actions
        inst_total += pomdp.priority_arr[inst]
    end
    println("inst usage: ",inst_total)
    println("reward: ", r_total)

    # Plotting 
    plot(
        plot(counts, rewards, label="rewards", linewidth=2,title="Rewards over 100 timesteps", xlabel="Timestep", ylabel="Reward"),
        plot(counts, battery_levels, label="battery level", linewidth=2,color=:green,title="Battery Level over 100 timesteps", xlabel="Timestep", ylabel="Battery",ylims=(0,100)),
        scatter(counts_a, actions, label="turned on", markersize=0.5,title="Instruments On", xlabel="Timestep", ylabel="Instrument ID",ylims=(0,6),color=:black),
        layout = (1, 3),  # Arrange subplots in 1 row and 2 columns
        size=(1200, 400),   # Set figure size
        suptitle=string("Value Iteration\n Priority Array: ", join(pomdp.priority_arr, ", ")," \n Instrument Battery Usage: ", join( pomdp.inst_battery_usage, ", "))
        )
    
end


function plot_deterministic_test(pomdp,deterministic_policy,start_state,verbose = true)

    # Initialize data for plotting
    rewards = Float64[]  # x-coordinates
    counts = Float64[]  # y-coordinates
    states = Float64[]
    battery_levels = Float64[]
    actions = Int[]
    counts_a = Int[]

    # Initialization of state 
    s = state_toindex(Int(start_state), 0, (2^pomdp.num_inst)) #rand(initialstate(pomdp))

    # initialization of simulation 
    r_total = 0.0
    d = 1

    # for 100 time steps
    for i in 0:100
        
        # generate an action given initial state
        a = deterministic_policy(pomdp,s)

        # progress to next state with action
        s, o, r = @gen(:sp,:o,:r)(pomdp, s, a)
        d *= discount(pomdp)

        # for debugging purposes
        if verbose
            println("  state:", s," obs:", o," reward:", r, " r_total: ",r_total," action:", a, " idle_times: ", pomdp.idle_times)
        end

        # All for plotting and saving data
        r_total += r
        battery_level, prev_action = state_decompose(2^pomdp.num_inst, s)
        push!(counts, i)
        push!(rewards, r_total) 
        push!(states, s)
        push!(battery_levels, battery_level)
        for (index,bool) in pairs(to_fixed_binary(a, pomdp.num_inst))
            if bool == 1
                push!(counts_a, i)
                push!(actions, index)
            end
        end
    end

    # Final statistics
    inst_total = 0
    for inst in actions
        inst_total += pomdp.priority_arr[inst]
    end
    println("inst usage: ",inst_total)
    println("reward: ", r_total)

    # Plotting 
    plot(
        plot(counts, rewards, label="rewards", linewidth=2,title="Rewards over 100 timesteps", xlabel="Timestep", ylabel="Reward"),
        plot(counts, battery_levels, label="battery level", linewidth=2,color=:green,title="Battery Level over 100 timesteps", xlabel="Timestep", ylabel="Battery",ylims=(0,100)),
        scatter(counts_a, actions, label="turned on", markersize=0.5,title="Instruments On", xlabel="Timestep", ylabel="Instrument ID",ylims=(0,6),color=:black),
        layout = (1, 3),  # Arrange subplots in 1 row and 2 columns
        size=(1300, 400),   # Set figure size
        suptitle=string("Deterministic\n Priority Array: ", join(pomdp.priority_arr, ", ")," \n Instrument Battery Usage: ", join( pomdp.inst_battery_usage, ", "))
        )
    
end

