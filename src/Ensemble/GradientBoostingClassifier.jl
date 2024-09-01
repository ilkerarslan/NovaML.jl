using Random
using Statistics
using Distributions

import ...NovaML: AbstractModel, sigmoid, logit
import ..Tree: DecisionTreeRegressor

mutable struct GradientBoostingClassifier <: AbstractModel
    loss::String
    learning_rate::Float64
    n_estimators::Int
    subsample::Float64
    criterion::String
    min_samples_split::Union{Int, Float64}
    min_samples_leaf::Union{Int, Float64}
    min_weight_fraction_leaf::Float64
    max_depth::Union{Int, Nothing}
    min_impurity_decrease::Float64
    init::Union{AbstractModel, String, Nothing}
    random_state::Union{Int, Nothing}
    max_features::Union{Int, Float64, String, Nothing}
    verbose::Int
    max_leaf_nodes::Union{Int, Nothing}
    warm_start::Bool
    validation_fraction::Float64
    n_iter_no_change::Union{Int, Nothing}
    tol::Float64
    ccp_alpha::Float64

    # Fitted attributes
    estimators_::Vector{Vector{DecisionTreeRegressor}}
    classes_::Vector
    n_classes_::Int
    feature_importances_::Union{Vector{Float64}, Nothing}
    oob_improvement_::Union{Vector{Float64}, Nothing}
    train_score_::Vector{Float64}
    n_estimators_::Int
    init_::Union{AbstractModel, Nothing}
    fitted::Bool

    function GradientBoostingClassifier(;
        loss::String="log_loss",
        learning_rate::Float64=0.1,
        n_estimators::Int=100,
        subsample::Float64=1.0,
        criterion::String="friedman_mse",
        min_samples_split::Union{Int, Float64}=2,
        min_samples_leaf::Union{Int, Float64}=1,
        min_weight_fraction_leaf::Float64=0.0,
        max_depth::Union{Int, Nothing}=3,
        min_impurity_decrease::Float64=0.0,
        init::Union{AbstractModel, String, Nothing}=nothing,
        random_state::Union{Int, Nothing}=nothing,
        max_features::Union{Int, Float64, String, Nothing}=nothing,
        verbose::Int=0,
        max_leaf_nodes::Union{Int, Nothing}=nothing,
        warm_start::Bool=false,
        validation_fraction::Float64=0.1,
        n_iter_no_change::Union{Int, Nothing}=nothing,
        tol::Float64=1e-4,
        ccp_alpha::Float64=0.0
    )
        new(
            loss, learning_rate, n_estimators, subsample, criterion,
            min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_depth, min_impurity_decrease, init, random_state,
            max_features, verbose, max_leaf_nodes, warm_start,
            validation_fraction, n_iter_no_change, tol, ccp_alpha,
            Vector{Vector{DecisionTreeRegressor}}(), [], 0, nothing, nothing,
            Float64[], 0, nothing, false
        )
    end
end

function (gbm::GradientBoostingClassifier)(X::AbstractMatrix, y::AbstractVector)
    n_samples, n_features = size(X)
    
    # Initialize random state
    if gbm.random_state !== nothing
        Random.seed!(gbm.random_state)
    end

    # Initialize classes and n_classes_
    gbm.classes_ = sort(unique(y))
    gbm.n_classes_ = length(gbm.classes_)

    # Initialize estimators
    if !gbm.warm_start || isempty(gbm.estimators_)
        gbm.estimators_ = [DecisionTreeRegressor[] for _ in 1:gbm.n_estimators]
    end

    # Initialize init estimator
    if gbm.init === nothing
        gbm.init_ = fit_initial_estimator(y)
    elseif isa(gbm.init, AbstractString) && gbm.init == "zero"
        gbm.init_ = ZeroEstimator()
    else
        gbm.init_ = gbm.init
    end

    # Initialize predictions
    y_pred = gbm.init_(X)

    # Main boosting loop
    for i in 1:gbm.n_estimators
        # Compute negative gradient
        negative_gradient = compute_negative_gradient(y, y_pred, gbm.loss)

        # Fit a regression tree to the negative gradient
        tree = DecisionTreeRegressor(
            max_depth=gbm.max_depth,
            min_samples_split=gbm.min_samples_split,
            min_samples_leaf=gbm.min_samples_leaf,
            min_weight_fraction_leaf=gbm.min_weight_fraction_leaf,
            max_features=gbm.max_features,
            max_leaf_nodes=gbm.max_leaf_nodes,
            min_impurity_decrease=gbm.min_impurity_decrease,
            random_state=gbm.random_state !== nothing ? rand(1:10000) : nothing
        )

        # Subsample if needed
        if gbm.subsample < 1.0
            sample_indices = rand(1:n_samples, round(Int, n_samples * gbm.subsample))
            X_subsample = X[sample_indices, :]
            negative_gradient_subsample = negative_gradient[sample_indices]
        else
            X_subsample = X
            negative_gradient_subsample = negative_gradient
        end

        tree(X_subsample, negative_gradient_subsample)
        push!(gbm.estimators_[i], tree)

        # Update predictions
        y_pred .+= gbm.learning_rate .* tree(X)

        # Store train score
        push!(gbm.train_score_, compute_loss(y, y_pred, gbm.loss))

        # Early stopping
        if gbm.n_iter_no_change !== nothing && i >= gbm.n_iter_no_change
            if all(abs.(diff(gbm.train_score_[end-gbm.n_iter_no_change+1:end])) .< gbm.tol)
                break
            end
        end
    end

    gbm.n_estimators_ = length(gbm.estimators_)
    gbm.feature_importances_ = compute_feature_importances(gbm)
    gbm.fitted = true
    return gbm
end

function (gbm::GradientBoostingClassifier)(X::AbstractMatrix; type=nothing)
    if !gbm.fitted
        throw(ErrorException("This GradientBoostingClassifier instance is not fitted yet. Call the model with training data before using it for predictions."))
    end

    # Get raw predictions (logits)
    raw_predictions = logit.(gbm.init_(X))
    for i in 1:gbm.n_estimators_
        raw_predictions .+= gbm.learning_rate .* gbm.estimators_[i][1](X)
    end

    if type == :probs
        return sigmoid.(raw_predictions)
    else
        return Int.(raw_predictions .>= 0)
    end
end

function compute_negative_gradient(y::AbstractVector, y_pred::AbstractVector, loss::String)
    if loss == "log_loss"
        return y .- sigmoid.(y_pred)
    elseif loss == "exponential"
        return y .* exp.(-y .* (2 .* sigmoid.(y_pred) .- 1))
    else
        throw(ArgumentError("Unsupported loss function: $loss"))
    end
end

function compute_loss(y::AbstractVector, y_pred::AbstractVector, loss::String)
    if loss == "log_loss"
        # Clip predictions to avoid log(0) or log(1)
        y_pred_clipped = clamp.(y_pred, 1e-15, 1 - 1e-15)
        return -mean(y .* log.(y_pred_clipped) .+ (1 .- y) .* log.(1 .- y_pred_clipped))
    elseif loss == "exponential"
        return mean(exp.(-y .* (2 .* y_pred .- 1)))
    else
        throw(ArgumentError("Unsupported loss function: $loss"))
    end
end

function compute_feature_importances(gbm::GradientBoostingClassifier)
    n_features = size(gbm.estimators_[1][1].feature_importances_, 1)
    importances = zeros(n_features)
    
    for estimators in gbm.estimators_
        for tree in estimators
            importances .+= tree.feature_importances_
        end
    end
    
    importances ./= gbm.n_estimators_
    return importances
end

struct InitialEstimator <: AbstractModel
    prob::Float64
end

function (estimator::InitialEstimator)(X::AbstractMatrix)
    return fill(estimator.prob, size(X, 1))
end

function fit_initial_estimator(y::AbstractVector)
    prob = mean(y .== 1)
    return InitialEstimator(prob)
end

struct ZeroEstimator <: AbstractModel end

(::ZeroEstimator)(X::AbstractMatrix) = zeros(size(X, 1))

function Base.show(io::IO, gbm::GradientBoostingClassifier)
    println(io, "GradientBoostingClassifier(")
    println(io, "  loss=$(gbm.loss),")
    println(io, "  learning_rate=$(gbm.learning_rate),")
    println(io, "  n_estimators=$(gbm.n_estimators),")
    println(io, "  subsample=$(gbm.subsample),")
    println(io, "  criterion=$(gbm.criterion),")
    println(io, "  min_samples_split=$(gbm.min_samples_split),")
    println(io, "  min_samples_leaf=$(gbm.min_samples_leaf),")
    println(io, "  min_weight_fraction_leaf=$(gbm.min_weight_fraction_leaf),")
    println(io, "  max_depth=$(gbm.max_depth),")
    println(io, "  min_impurity_decrease=$(gbm.min_impurity_decrease),")
    println(io, "  init=$(gbm.init),")
    println(io, "  random_state=$(gbm.random_state),")
    println(io, "  max_features=$(gbm.max_features),")
    println(io, "  verbose=$(gbm.verbose),")
    println(io, "  max_leaf_nodes=$(gbm.max_leaf_nodes),")
    println(io, "  warm_start=$(gbm.warm_start),")
    println(io, "  validation_fraction=$(gbm.validation_fraction),")
    println(io, "  n_iter_no_change=$(gbm.n_iter_no_change),")
    println(io, "  tol=$(gbm.tol),")
    println(io, "  ccp_alpha=$(gbm.ccp_alpha),")
    println(io, "  fitted=$(gbm.fitted)")
    print(io, ")")
end