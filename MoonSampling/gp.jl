using GaussianProcesses, Plots

# Create empty arrays for initialization
x = Matrix{Float64}(undef, 2, 0)  # 2D input space, no points yet
y = Float64[]  # Empty array for outputs

# Define mean and kernel functions
mZero = MeanZero()                # Zero mean function
kern = SE([0.0, 0.0], 0.0)       # SE kernel with 2D length scales and amplitude

# Create an elastic Gaussian process
gp = ElasticGPE(x, y, mZero, kern)

# Create a 5x5 grid with spacing of 1 from 0 to 5
x_range = 0:1:4
y_range = 0:1:4
x_coords = vec(repeat(x_range, inner=length(y_range)))
y_coords = vec(repeat(y_range, outer=length(x_range)))

# Format as 2×25 matrix (2 dimensions, 25 points)
x = Float64.(vcat(x_coords', y_coords'))  # Stack coordinates as rows

# Create example outputs
truth_map = rand([-2.0, 2.0], 5, 5)
y = vec(truth_map)
y = 1 ./ (1 .+ exp.(-y))  # logistic transform
append!(gp, x, y)

# # Make predictions at new points
# x_test = [2.5 3.5;       # Test points x coordinates
#           1.0 2.0]       # Test points y coordinates
# μ, σ² = predict_y(gp, x_test)
# print(μ, σ²)

p1 = heatmap(truth_map, color=:blues, legend=false, axis=false, grid=false, aspect_ratio=:equal)
p2 = wireframe(gp, color=:blues)
plot(p1, p2)