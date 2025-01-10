using POMDPs, GaussianProcesses, AbstractGPs

@with_kw mutable struct Surrogate{F<:Union{AbstractGPs.AbstractGP, AbstractGPs.PosteriorGP, GaussianProcesses.GPE, Nothing}}
    f::F
    x = []
    y = []
    sig = exp(-0.1) # signal variance
end

struct GPUpdater <: POMDPs.Updater end

struct GPBelief
    parameters_to_gaussian_process
end

POMDPs.initialize_belief(up::GPUpdater, d) = initialize_gp() # takes updater, state dist.
# return surrogate model, GP with empty kernel (all values 0.5)

function POMDPs.update(up::GPUpdater, b::GPBelief, a::Int, o::Bool)
    if a in 3:pomdp.map_size[1]*pomdp.map_size[2]+2 # scoop
        new_pos = (collect(Tuple(CartesianIndices(pomdp.map_size)[a]))) # convert a to cartesian index
    bp = copy(b)
    # get x_obs, y_obs from a, o
    X = hcat(gp.x, x_obs)
    y_obs = # System.evaluate(sparams, inputs; subdir=subdir) ???
    Y = vcat(gp.y, y_obs)
    gp = gp_fit(gp, X, Y)
    return bp
end

#################################

# scale GP to [0, 1]
logit(y; s=1/10) = log(y / (1 - y)) / s 
apply(y) = logit(transform(y))
apply(y::Array) = apply.(y)

function initialize_gp(; sig=exp(-0.1), l=exp(-0.1))
    kernel = sig^2 * IdentityKernel() # choose white noise kernel
    mean_f = AbstractGPs.ZeroMean()
    return Surrogate(; f=AbstractGPs.GP(mean_f, kernel), sig) # uniform, flat GP (values == 0.5?)
end

function gp_fit(::Union{Surrogate{<:GaussianProcesses.GPE}, Nothing}, X, Y; nu=1/2, ll=-0.1, l_sig=-0.1, opt=false)
    # re-fit surrogate model
    # X = inputs
    # Y = outputs
    # nu, ll, l_sig = kernel parameters
    # opt = optimize GP hyperparameters?
    kernel = Matern(nu, ll, l_sig)
    mean_f = MeanZero()
    Z = apply.(Y)
    f = GaussianProcesses.GP(X, Z, mean_f, kernel)
    gp = Surrogate(; f, x=X, y=Z)
    opt && @suppress optimize!(gp, method=NelderMead())
    return gp
end

"""
Run a single input sample through the system, re-fit surrogate model.
"""
function run_single_sample(gp, models, sparams, sample; subdir="test")
    input = System.generate_input(sparams, sample; models, subdir) # do i want to generate input? 
    inputs = [input]
    X = hcat(gp.x, sample)
    Yp = System.evaluate(sparams, inputs; subdir=subdir) # how to get output of GP?
    Y = vcat(gp.y, Yp)
    gp = gp_fit(gp, X, Y)
    display(plot_soft_boundary(gp, models))
    return gp
end