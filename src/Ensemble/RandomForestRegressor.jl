using Random
using Statistics: mean
using DataStructures: DefaultDict

import ...NovaML: AbstractModel
import ..Tree: DecisionTreeRegressor

"""
    RandomForestRegressor <: AbstractModel

A random forest regressor.

Random forests are an ensemble learning method for regression that operate by constructing
a multitude of decision trees at training time and outputting the mean prediction of the
individual trees.

# Fields
- `n_estimators::Int`: The number of trees in the forest.
- `criterion::String`: The function to measure the quality of a split.
- `max_depth::Union{Int, Nothing}`: The maximum depth of the tree.
- `min_samples_split::Int`: The minimum number of samples required to split an internal node.
- `min_samples_leaf::Int`: The minimum number of samples required to be at a leaf node.
- `min_weight_fraction_leaf::Float64`: The minimum weighted fraction of the sum total of weights required to be at a leaf node.
- `max_features::Union{Int, Float64, String, Nothing}`: The number of features to consider when looking for the best split.
- `max_leaf_nodes::Union{Int, Nothing}`: Grow trees with max_leaf_nodes in best-first fashion.
- `min_impurity_decrease::Float64`: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
- `bootstrap::Bool`: Whether bootstrap samples are used when building trees.
- `oob_score::Bool`: Whether to use out-of-bag samples to estimate the generalization score.
- `n_jobs::Union{Int, Nothing}`: The number of jobs to run in parallel.
- `random_state::Union{Int, Nothing}`: Controls both the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node.
- `verbose::Int`: Controls the verbosity when fitting and predicting.
- `warm_start::Bool`: When set to true, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.
- `ccp_alpha::Float64`: Complexity parameter used for Minimal Cost-Complexity Pruning.
- `max_samples::Union{Int, Float64, Nothing}`: If bootstrap is True, the number of samples to draw from X to train each base estimator.

# Example
```julia
rf = RandomForestRegressor(n_estimators=100, max_depth=10)
rf(X, y)  # Fit the model
predictions = rf(X_test)  # Make predictions
```
"""
mutable struct RandomForestRegressor <: AbstractModel
    n_estimators::Int
    criterion::String
    max_depth::Union{Int, Nothing}
    min_samples_split::Int
    min_samples_leaf::Int
    min_weight_fraction_leaf::Float64
    max_features::Union{Int, Float64, String, Nothing}
    max_leaf_nodes::Union{Int, Nothing}
    min_impurity_decrease::Float64
    bootstrap::Bool
    oob_score::Bool
    n_jobs::Union{Int, Nothing}
    random_state::Union{Int, Nothing}
    verbose::Int
    warm_start::Bool
    ccp_alpha::Float64
    max_samples::Union{Int, Float64, Nothing}
    trees::Vector{DecisionTreeRegressor}
    feature_importances_::Union{Vector{Float64}, Nothing}
    n_features_in_::Int
    feature_names_in_::Vector{String}
    n_outputs_::Int
    oob_score_::Union{Float64, Nothing}
    oob_prediction_::Union{Vector{Float64}, Nothing}
    fitted::Bool

    function RandomForestRegressor(;
        n_estimators::Int=100,
        criterion::String="squared_error",
        max_depth::Union{Int, Nothing}=nothing,
        min_samples_split::Int=2,
        min_samples_leaf::Int=1,
        min_weight_fraction_leaf::Float64=0.0,
        max_features::Union{Int, Float64, String, Nothing}=1.0,
        max_leaf_nodes::Union{Int, Nothing}=nothing,
        min_impurity_decrease::Float64=0.0,
        bootstrap::Bool=true,
        oob_score::Bool=false,
        n_jobs::Union{Int, Nothing}=nothing,
        random_state::Union{Int, Nothing}=nothing,
        verbose::Int=0,
        warm_start::Bool=false,
        ccp_alpha::Float64=0.0,
        max_samples::Union{Int, Float64, Nothing}=nothing
    )
        new(
            n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf,
            min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease,
            bootstrap, oob_score, n_jobs, random_state, verbose, warm_start, ccp_alpha,
            max_samples, DecisionTreeRegressor[], nothing, 0, String[], 0, nothing, nothing, false
        )
    end
end

"""
    (forest::RandomForestRegressor)(X::AbstractMatrix, y::AbstractVector)
Fit the random forest regressor.

# Arguments
- `X::AbstractMatrix`: The input samples.
- `y::AbstractVector`: The target values.

# Returns
- `RandomForestRegressor`: The fitted model.
"""
function (forest::RandomForestRegressor)(X::AbstractMatrix, y::AbstractVector)
    n_samples, n_features = size(X)
    forest.n_features_in_ = n_features
    forest.n_outputs_ = 1

    if forest.random_state !== nothing
        Random.seed!(forest.random_state)
    end

    max_features = get_max_features(forest, n_features)

    forest.trees = []
    feature_importances = zeros(n_features)

    for _ in 1:forest.n_estimators
        tree = DecisionTreeRegressor(
            criterion=forest.criterion,
            max_depth=forest.max_depth,
            min_samples_split=forest.min_samples_split,
            min_samples_leaf=forest.min_samples_leaf,
            min_weight_fraction_leaf=forest.min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=forest.max_leaf_nodes,
            min_impurity_decrease=forest.min_impurity_decrease,
            random_state=forest.random_state !== nothing ? rand(1:10000) : nothing
        )

        if forest.bootstrap
            indices = rand(1:n_samples, n_samples)
            X_bootstrap, y_bootstrap = X[indices, :], y[indices]
        else
            X_bootstrap, y_bootstrap = X, y
        end

        tree(X_bootstrap, y_bootstrap)
        push!(forest.trees, tree)
        
        # Update feature importances
        if tree.feature_importances_ !== nothing
            feature_importances .+= tree.feature_importances_
        end
    end

    # Average and normalize feature importances
    if !all(feature_importances .== 0)
        forest.feature_importances_ = feature_importances ./ forest.n_estimators
        forest.feature_importances_ ./= sum(forest.feature_importances_)
    else
        forest.feature_importances_ = nothing
    end

    if forest.oob_score
        forest.oob_score_, forest.oob_prediction_ = compute_oob_score(forest, X, y)
    end

    forest.fitted = true
    return forest
end

"""
    (forest::RandomForestRegressor)(X::AbstractMatrix)
Predict regression target for X.

# Arguments
- `X::AbstractMatrix`: The input samples.

# Returns
- `Vector{Float64}`: The predicted values.
"""
function (forest::RandomForestRegressor)(X::AbstractMatrix)
    if !forest.fitted
        throw(ErrorException("This RandomForestRegressor instance is not fitted yet. Call the model with training data before using it for predictions."))
    end

    n_samples = size(X, 1)
    predictions = zeros(n_samples)

    for tree in forest.trees
        predictions .+= tree(X)
    end

    return predictions ./ forest.n_estimators
end

"""
get_max_features(forest::RandomForestRegressor, n_features::Int)
Get the number of features to consider when looking for the best split.

# Arguments
- `forest::RandomForestRegressor`: The random forest regressor.
- `n_features::Int`: The total number of features.

# Returns
# `Int`: The number of features to consider.
"""
function get_max_features(forest::RandomForestRegressor, n_features::Int)
    if forest.max_features === nothing || forest.max_features == 1.0
        return n_features
    elseif isa(forest.max_features, Int)
        return min(forest.max_features, n_features)
    elseif isa(forest.max_features, Float64)
        return round(Int, forest.max_features * n_features)
    elseif forest.max_features == "sqrt"
        return round(Int, sqrt(n_features))
    elseif forest.max_features == "log2"
        return round(Int, log2(n_features))
    else
        throw(ArgumentError("Invalid max_features parameter"))
    end
end

"""
    compute_oob_score(forest::RandomForestRegressor, X::AbstractMatrix, y::AbstractVector)
Compute out-of-bag (OOB) score for the random forest regressor.

# Arguments
- `forest::RandomForestRegressor`: The random forest regressor.
- `X::AbstractMatrix`: The input samples.
- `y::AbstractVector`: The target values.

# Returns
- `Tuple{Float64, Vector{Float64}}`: The OOB score and OOB predictions.
"""
function compute_oob_score(forest::RandomForestRegressor, X::AbstractMatrix, y::AbstractVector)
    n_samples = size(X, 1)
    oob_predictions = zeros(n_samples)
    n_predictions = zeros(Int, n_samples)

    for (i, tree) in enumerate(forest.trees)
        oob_mask = .!tree.estimators_samples_
        oob_predictions[oob_mask] .+= tree(X[oob_mask, :])
        n_predictions[oob_mask] .+= 1
    end

    oob_score = mean((y[n_predictions .> 0] .- oob_predictions[n_predictions .> 0] ./ n_predictions[n_predictions .> 0]).^2)
    return 1 - oob_score, oob_predictions ./ max.(n_predictions, 1)
end

"""
    Base.show(io::IO, forest::RandomForestRegressor)
Custom show method for RandomForestRegressor.

# Arguments
- `io::IO`: The I/O stream.
- `forest::RandomForestRegressor`: The random forest regressor to display.
"""
function Base.show(io::IO, forest::RandomForestRegressor)
    println(io, "RandomForestRegressor(")
    println(io, "  n_estimators=$(forest.n_estimators),")
    println(io, "  criterion=\"$(forest.criterion)\",")
    println(io, "  max_depth=$(forest.max_depth),")
    println(io, "  min_samples_split=$(forest.min_samples_split),")
    println(io, "  min_samples_leaf=$(forest.min_samples_leaf),")
    println(io, "  min_weight_fraction_leaf=$(forest.min_weight_fraction_leaf),")
    println(io, "  max_features=$(forest.max_features),")
    println(io, "  max_leaf_nodes=$(forest.max_leaf_nodes),")
    println(io, "  min_impurity_decrease=$(forest.min_impurity_decrease),")
    println(io, "  bootstrap=$(forest.bootstrap),")
    println(io, "  oob_score=$(forest.oob_score),")
    println(io, "  n_jobs=$(forest.n_jobs),")
    println(io, "  random_state=$(forest.random_state),")
    println(io, "  verbose=$(forest.verbose),")
    println(io, "  warm_start=$(forest.warm_start),")
    println(io, "  ccp_alpha=$(forest.ccp_alpha),")
    println(io, "  max_samples=$(forest.max_samples),")
    println(io, "  fitted=$(forest.fitted)")
    print(io, ")")
end