using Distances
using Statistics

import ...NovaML: AbstractModel

mutable struct KNeighborsRegressor <: AbstractModel
    n_neighbors::Int
    weights::Symbol
    algorithm::Symbol
    leaf_size::Int
    p::Real
    metric::Metric
    metric_params::Union{Dict, Nothing}
    n_jobs::Union{Int, Nothing}

    # Attributes
    n_features_in_::Int
    n_samples_fit_::Int

    # Internal data
    X::Union{Matrix{Float64}, Nothing}
    y::Union{Vector{Float64}, Nothing}

    fitted::Bool

    function KNeighborsRegressor(;
        n_neighbors::Int = 5,
        weights::Symbol = :uniform,
        algorithm::Symbol = :auto,
        leaf_size::Int = 30,
        p::Real = 2,
        metric::Union{String, Metric} = "minkowski",
        metric_params::Union{Dict, Nothing} = nothing,
        n_jobs::Union{Int, Nothing} = nothing
    )
        @assert n_neighbors > 0 "n_neighbors must be positive"
        @assert weights in [:uniform, :distance] "weights must be :uniform or :distance"
        @assert algorithm in [:auto, :brute] "Only :auto and :brute algorithms are currently supported"
        @assert leaf_size > 0 "leaf_size must be positive"
        @assert p > 0 "p must be positive"

        if typeof(metric) == String
            if metric == "minkowski"
                metric = Minkowski(p)
            elseif metric == "euclidean"
                metric = Euclidean()
            elseif metric == "manhattan"
                metric = Cityblock()
            else
                error("Unsupported metric string: $metric")
            end
        end

        new(
            n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs,
            0, 0, nothing, nothing, false
        )
    end
end

function (reg::KNeighborsRegressor)(X::Matrix{Float64}, y::Vector)
    reg.X = X
    reg.y = Float64.(y)
    reg.n_features_in_ = size(X, 2)
    reg.n_samples_fit_ = size(X, 1)

    reg.fitted = true
    return reg
end

function (reg::KNeighborsRegressor)(X::Matrix{Float64})
    if !reg.fitted
        throw(ErrorException("This KNeighborsRegressor instance is not fitted yet. Call with training data before using it for predictions."))
    end

    n_samples = size(X, 1)
    predictions = Vector{Float64}(undef, n_samples)

    for i in 1:n_samples
        distances = [evaluate(reg.metric, X[i, :], reg.X[j, :]) for j in 1:reg.n_samples_fit_]
        sorted_indices = sortperm(distances)
        neighbor_indices = sorted_indices[1:reg.n_neighbors]
        neighbor_distances = distances[neighbor_indices]
        neighbor_targets = reg.y[neighbor_indices]

        if reg.weights == :uniform
            predictions[i] = mean(neighbor_targets)
        else # distance weighting
            w = 1.0 ./ (neighbor_distances .+ eps())
            predictions[i] = sum(w .* neighbor_targets) / sum(w)
        end
    end

    return predictions
end

function Base.show(io::IO, reg::KNeighborsRegressor)
    print(io, "KNeighborsRegressor(n_neighbors=$(reg.n_neighbors), ",
        "weights=$(reg.weights), algorithm=$(reg.algorithm), ",
        "leaf_size=$(reg.leaf_size), p=$(reg.p), metric=$(reg.metric), ",
        "metric_params=$(reg.metric_params), n_jobs=$(reg.n_jobs), fitted=$(reg.fitted))")
end
