using Distributions, Random, Plots, LinearAlgebra

"""
Functions for generating Gaussian Mixture Models to initialize and update belief.
"""

function GaussianMixtureModel(x, y, n)
    weights = rand(Dirichlet(n, 1.0)) # mixture weights (positive, sums to 1)
    mus = [rand(Uniform(1, x), 2) .* [1, y/x] for _ in 1:n] # list of 2D means
    covs = GenerateCovs(n) # list of 2x2 covariance matrices

    components = [MvNormal(mu, cov) for (mu, cov) in zip(mus, covs)]
    return MixtureModel(components, weights)
end

function GenerateCovs(n)
    covs = Vector{Matrix{Float64}}(undef, n) # unilitialized
    for i in 1:n
        diag = rand(Uniform(0.3, 0.9))
        C = [diag 0.; 0. diag]
        covs[i] = C # must be positive semi-definite
    end
    return covs
end

function AddGaussian(gmm, x, y, new_weight=0.2)
    new_dist = MvNormal([x, y], [1. 0.; 0. 1.]) # spatial correlation? pos + negative?
    new_dists = copy(components(gmm))
    new_weights = copy(probs(gmm))
    push!(new_dists, new_dist)
    push!(new_weights, new_weight)
    new_weights /= sum(new_weights) # normalize
    return MixtureModel(new_dists, new_weights)
end

function PlotGMM(gmm, x, y, n, res=100)
    x_vals = range(1, x, length=res)
    y_vals = range(1, y, length=res)
    density = zeros(res, res)

    for i in 1:res, j in 1:res
        point = [x_vals[i], y_vals[j]]
        density[j, i] = pdf(gmm, point)
    end
    Plots.surface(x_vals, y_vals, density, title="2D GMM Density Surface", xlabel="x", ylabel="y", zlabel="Density")
end

gmm = GaussianMixtureModel(5, 5, 4)
PlotGMM(gmm, 5, 5, 4)
# gmm = AddGaussian(gmm, 2, 3)
# PlotGMM(gmm, 5, 5, 5)
