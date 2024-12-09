
struct GPUpdater <: POMDPs.Updater end

struct GPBelief
    gp::Any # ElasticGPE
end

Observations
POMDPs.observations(mdp::ExtractionMDP) = [true, false]

function POMDPs.observation(mdp::ExtractionMDP, s::ExtractionState, a::Int, sp::ExtractionState)
    if (a == 1 || a ==2) && s.full
        if pomdp.truth_map[s.pos] == true # good sample
            return SparseCat([true, false], [1.0, 0.0]) # [1, 0] = deterministic
        else # bad sample
            return SparseCat([true, false], [0.0, 1.0]) # [0, 1] = deterministic
        end
    end
    return None # no information gathered
end

function POMDPs.obsindex(mdp::ExtractionMDP, o::Bool)
    return o ? 1 : 2
end

Belief
function POMDPs.initialize_belief(up::GPUpdater, d) # d = initial state dist. (unused)
    # create an empty elastic Gaussian process
    x = Matrix{Float64}(undef, 2, 0)  # 2D input space, no points yet
    y = Float64[]                     # empty output array
    mean = MeanZero()                 # zero mean function
    kern = SE([0.0, 0.0], 0.0)        # SE kernel with 2D length scales and amplitude
    gp = ElasticGPE(x, y, mean, kern)

    # Create an mxm grid with spacing of 1
    x_range = 0:1:mdp.map_size[1]-1
    y_range = 0:1:mdp.map_size[2]-1
    x_coords = vec(repeat(x_range, inner=length(y_range)))
    y_coords = vec(repeat(y_range, outer=length(x_range)))
    x = Float64.(vcat(x_coords', y_coords'))  # Format as 2×nm matrix (2 dimensions, nm points)
    y = vec(mdp.truth_map)                    # Create outputs
    y = 1 ./ (1 .+ exp.(-y))                  # logistic transform (scale to [0,1])
    append!(gp, x, y)
    b = GPBelief(gp)
    return b
end

function POMDPs.update(up::GPUpdater, b::GPBelief, a::Int, o::Bool)
    bp = copy(b)
    new_x = [s.pos[1]; s.pos[2]]  # 2D coordinates as column vector
    new_y = o == true ? 1.0 : 0 # good or bad sample
    append!(bp.gp, new_x, new_y)
    return bp
end