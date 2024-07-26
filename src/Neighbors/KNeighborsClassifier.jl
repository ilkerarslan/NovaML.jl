using Distances
using Statistics
using StatsBase: mode, sample, Weights

import ...NovaML: AbstractModel

mutable struct KNeighborsClassifier <: AbstractModel
    n_neighbors::Int
    weights::Symbol
    algorithm::Symbol
    leaf_size::Int
    p::Real
    metric::Metric
    metric_params::Union{Dict, Nothing}
    n_jobs::Union{Int, Nothing}
    
    # Attributes
    classes_::Vector{Any}
    n_features_in_::Int
    n_samples_fit_::Int
    outputs_2d_::Bool
    
    # Internal data
    X::Union{Matrix{Float64}, Nothing}
    y::Union{Vector, Nothing}
    
    fitted::Bool

    function KNeighborsClassifier(;
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
            Vector{Any}(), 0, 0, false, nothing, nothing, false
        )
    end
end

function (clf::KNeighborsClassifier)(X::Matrix{Float64}, y::Vector)
    clf.X = X
    clf.y = y
    clf.n_features_in_ = size(X, 2)
    clf.n_samples_fit_ = size(X, 1)
    clf.classes_ = unique(y)
    clf.outputs_2d_ = false  # Assuming 1D output for now

    clf.fitted = true
    return clf
end

function (clf::KNeighborsClassifier)(X::Matrix{Float64}; type=nothing)
    if !clf.fitted
        throw(ErrorException("This KNeighborsClassifier instance is not fitted yet. Call with training data before using it for predictions."))
    end

    n_samples = size(X, 1)
    predictions = Vector{eltype(clf.y)}(undef, n_samples)

    if type == :probs
        probas = zeros(Float64, n_samples, length(clf.classes_))
    end

    for i in 1:n_samples
        distances = [evaluate(clf.metric, X[i, :], clf.X[j, :]) for j in 1:clf.n_samples_fit_]
        sorted_indices = sortperm(distances)
        neighbor_indices = sorted_indices[1:clf.n_neighbors]
        neighbor_distances = distances[neighbor_indices]

        if clf.weights == :uniform
            if type == :probs
                for (idx, class) in enumerate(clf.classes_)
                    probas[i, idx] = count(==(class), clf.y[neighbor_indices]) / clf.n_neighbors
                end
            else
                predictions[i] = mode(clf.y[neighbor_indices])
            end
        else # distance weighting
            weights = 1 ./ (neighbor_distances .+ eps())
            if type == :probs
                for (idx, class) in enumerate(clf.classes_)
                    probas[i, idx] = sum(weights[clf.y[neighbor_indices] .== class]) / sum(weights)
                end
            else
                predictions[i] = sample(clf.y[neighbor_indices], Weights(weights))
            end
        end
    end

    if type == :probs
        return probas
    else
        return predictions
    end
end

function Base.show(io::IO, clf::KNeighborsClassifier)
    print(io, "KNeighborsClassifier(n_neighbors=$(clf.n_neighbors), ",
        "weights=$(clf.weights), algorithm=$(clf.algorithm), ",
        "leaf_size=$(clf.leaf_size), p=$(clf.p), metric=$(clf.metric), ",
        "metric_params=$(clf.metric_params), n_jobs=$(clf.n_jobs), fitted=$(clf.fitted))")
end