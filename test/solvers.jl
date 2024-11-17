include("../src/ActiveSampling.jl")
using .ActiveSampling
using POMDPs
using SARSOP
export pomdp

pomdp = CreateSamplePOMDP() # using default values
solver = SARSOPSolver()
policy = solve(solver, pomdp)

@show_requirements POMDPs.solve(solver, pomdp)