using Random
using Statistics
using Distributions
import ...NovaML: AbstractModel
import ...Tree: DecisionTreeClassifier

"""
    AdaBoostClassifier <: AbstractModel

An AdaBoost classifier.

An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset
and then fits additional copies of the classifier on the same dataset but where the weights of
incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

# Fields
- `base_estimator::Any`: The base estimator from which the boosted ensemble is built.
- `n_estimators::Int`: The maximum number of estimators at which boosting is terminated.
- `learning_rate::Float64`: Weight applied to each classifier at each boosting iteration.
- `algorithm::Symbol`: The SAMME algorithm to use when fitting the model.
- `random_state::Union{Int, Nothing}`: Controls the random seed given at each `base_estimator` at each boosting iteration.

# Fitted Attributes
- `estimators_::Vector{Any}`: The collection of fitted sub-estimators.
- `estimator_weights_::Vector{Float64}`: Weights for each estimator in the boosted ensemble.
- `estimator_errors_::Vector{Float64}`: Classification error for each estimator in the boosted ensemble.
- `classes_::Vector{Any}`: The classes labels.
- `n_classes_::Int`: The number of classes.
- `feature_importances_::Union{Vector{Float64}, Nothing}`: The feature importances if supported by the `base_estimator`.
- `fitted::Bool`: Whether the model has been fitted.

# Example
```julia
model = AdaBoostClassifier(n_estimators=100, learning_rate=1.0)
model(X, y)  # Fit the model
predictions = model(X_test)  # Make predictions
probabilities = model(X_test, type=:probs)  # Get probability estimates
```
"""
mutable struct AdaBoostClassifier <: AbstractModel
    base_estimator::Any
    n_estimators::Int
    learning_rate::Float64
    algorithm::Symbol
    random_state::Union{Int, Nothing}

    # Fitted attributes
    estimators_::Vector{Any}
    estimator_weights_::Vector{Float64}
    estimator_errors_::Vector{Float64}
    classes_::Vector{Any}
    n_classes_::Int
    feature_importances_::Union{Vector{Float64}, Nothing}
    fitted::Bool

    function AdaBoostClassifier(;
        base_estimator=nothing,
        n_estimators::Int=50,
        learning_rate::Float64=1.0,
        algorithm::Symbol=:SAMME,
        random_state::Union{Int, Nothing}=nothing
    )
        if base_estimator === nothing
            base_estimator = DecisionTreeClassifier(max_depth=1)
        end
        
        @assert algorithm in [:SAMME, :SAMME_R] "algorithm must be either :SAMME or :SAMME_R"
        @assert n_estimators > 0 "n_estimators must be positive"
        @assert learning_rate > 0 "learning_rate must be positive"

        new(
            base_estimator, n_estimators, learning_rate, algorithm, random_state,
            [], [], [], [], 0, nothing, false
        )
    end
end

"""
(model::AdaBoostClassifier)(X::AbstractMatrix, y::AbstractVector)
Fit the AdaBoost model.

# Arguments
- `X::AbstractMatrix`: The input samples.
- `y::AbstractVector`: The target values (class labels).

# Returns
- `AdaBoostClassifier`: The fitted model.
"""
function (model::AdaBoostClassifier)(X::AbstractMatrix, y::AbstractVector)
    if model.random_state !== nothing
        Random.seed!(model.random_state)
    end

    n_samples, n_features = size(X)
    model.classes_ = sort(unique(y))
    model.n_classes_ = length(model.classes_)
    
    class_to_index = Dict(class => i for (i, class) in enumerate(model.classes_))
    y_encoded = [class_to_index[label] for label in y]

    sample_weight = fill(1.0 / n_samples, n_samples)
    
    model.estimators_ = []
    model.estimator_weights_ = Float64[]
    model.estimator_errors_ = Float64[]

    for _ in 1:model.n_estimators
        estimator = deepcopy(model.base_estimator)
        
        # Train the estimator
        estimator(X, y, sample_weight=sample_weight)
        
        # Make predictions
        predictions = estimator(X)
        
        # Calculate error
        incorrect = predictions .!= y
        estimator_error = sum(sample_weight .* incorrect) / sum(sample_weight)

        # Check if the estimator is better than random guessing
        if estimator_error <= 0
            model.estimator_weights_ = [1.0]
            model.estimators_ = [estimator]
            break
        elseif estimator_error >= 0.5
            continue
        end

        # Calculate estimator weight
        beta = estimator_error / (1.0 - estimator_error)
        estimator_weight = model.learning_rate * log(1.0 / beta)

        # Store the estimator and its weight
        push!(model.estimators_, estimator)
        push!(model.estimator_weights_, estimator_weight)
        push!(model.estimator_errors_, estimator_error)

        # Update sample weights
        if model.algorithm == :SAMME
            sample_weight .*= exp.(estimator_weight .* incorrect)
        else # SAMME.R
            y_probability = estimator(X, type=:probs)
            sample_weight .*= exp.(-estimator_weight .* y_probability[CartesianIndex.(1:n_samples, y_encoded)])
        end

        # Normalize sample weights
        sample_weight ./= sum(sample_weight)
    end

    if !isempty(model.estimators_)
        model.feature_importances_ = _compute_feature_importances(model)
    else
        model.feature_importances_ = nothing
    end

    model.fitted = true
    return model
end

"""
    (model::AdaBoostClassifier)(X::AbstractMatrix; type=nothing)
Predict using the AdaBoost model.

# Arguments
- `X::AbstractMatrix`: The input samples.
- `type`: If set to :probs, return probability estimates for each class.

# Returns
- If type is :probs, returns probabilities of each class.
- Otherwise, returns predicted class labels.
"""
function (model::AdaBoostClassifier)(X::AbstractMatrix; type=nothing)
    if !model.fitted
        throw(ErrorException("This AdaBoostClassifier instance is not fitted yet. Call the model with training data before using it for predictions."))
    end

    n_samples = size(X, 1)
    
    if model.algorithm == :SAMME_R
        proba = zeros(n_samples, model.n_classes_)
        for (estimator, weight) in zip(model.estimators_, model.estimator_weights_)
            current_proba = estimator(X, type=:probs)
            proba .+= weight .* (log.(current_proba .+ eps()) .- log(1.0 / model.n_classes_))
        end
        proba = exp.(proba)
        proba ./= sum(proba, dims=2)
    else # SAMME
        proba = zeros(n_samples, model.n_classes_)
        for (estimator, weight) in zip(model.estimators_, model.estimator_weights_)
            predictions = estimator(X)
            for (i, pred) in enumerate(predictions)
                proba[i, findfirst(==(pred), model.classes_)] += weight
            end
        end
        proba ./= sum(model.estimator_weights_)
    end

    if type == :probs
        return proba
    else
        return [model.classes_[argmax(proba[i, :])] for i in 1:n_samples]
    end
end

"""
    _compute_feature_importances(model::AdaBoostClassifier)
Compute feature importances for the AdaBoost model.

# Arguments
- `model::AdaBoostClassifier`: The fitted AdaBoost model.

# Returns
- `Union{Vector{Float64}, Nothing}`: The feature importances if available, otherwise nothing.
"""
function _compute_feature_importances(model::AdaBoostClassifier)
    if isempty(model.estimators_)
        return nothing
    end

    if !all(hasfield.(typeof.(model.estimators_), :feature_importances_))
        return nothing
    end
    
    if any(est -> est.feature_importances_ === nothing, model.estimators_)
        return nothing
    end

    n_features = length(model.estimators_[1].feature_importances_)
    feature_importances = zeros(n_features)
    total_weight = sum(model.estimator_weights_)
    
    for (estimator, weight) in zip(model.estimators_, model.estimator_weights_)
        feature_importances .+= weight .* estimator.feature_importances_
    end
    
    return feature_importances ./ total_weight
end

"""
    Base.show(io::IO, model::AdaBoostClassifier)
Custom show method for AdaBoostClassifier.

# Arguments
- `io::IO`: The I/O stream.
- `model`::AdaBoostClassifier: The AdaBoost model to display.
"""
function Base.show(io::IO, model::AdaBoostClassifier)
    println(io, "AdaBoostClassifier(")
    println(io, "  base_estimator=$(model.base_estimator),")
    println(io, "  n_estimators=$(model.n_estimators),")
    println(io, "  learning_rate=$(model.learning_rate),")
    println(io, "  algorithm=$(model.algorithm),")
    println(io, "  random_state=$(model.random_state),")
    print(io, "  fitted=$(model.fitted)")
    print(io, ")")
end

"""
    get_params(model::AdaBoostClassifier; deep=true)
Get parameters for this estimator.

# Arguments
- `model::AdaBoostClassifier`: The AdaBoost model.
- `deep::Bool`: If true, will return the parameters for this estimator and contained subobjects that are estimators.

# Returns
- `Dict`: Parameter names mapped to their values.
"""
function get_params(model::AdaBoostClassifier; deep=true)
    params = Dict(
        :base_estimator => model.base_estimator,
        :n_estimators => model.n_estimators,
        :learning_rate => model.learning_rate,
        :algorithm => model.algorithm,
        :random_state => model.random_state
    )
    if deep && typeof(model.base_estimator) <: AbstractModel
        base_params = get_params(model.base_estimator, deep=true)
        for (key, value) in base_params
            params[Symbol("base_estimator__$key")] = value
        end
    end
    return params
end

"""
    set_params!(model::AdaBoostClassifier; kwargs...)
Set the parameters of this estimator.

# Arguments
- `model::AdaBoostClassifier`: The AdaBoost model.
- `kwargs...`: Estimator parameters.

# Returns
- `AdaBoostClassifier`: The estimator instance.
"""
function set_params!(model::AdaBoostClassifier; kwargs...)
    for (key, value) in kwargs
        if key in [:base_estimator, :n_estimators, :learning_rate, :algorithm, :random_state]
            setfield!(model, key, value)
        elseif startswith(string(key), "base_estimator__")
            base_key = Symbol(split(string(key), "__")[2])
            if typeof(model.base_estimator) <: AbstractModel
                set_params!(model.base_estimator; Dict(base_key => value)...)
            else
                @warn "Cannot set parameter $key for base_estimator of type $(typeof(model.base_estimator))"
            end
        else
            @warn "Invalid parameter $key for AdaBoostClassifier"
        end
    end
    return model
end

"""
    decision_function(model::AdaBoostClassifier, X::AbstractMatrix)
Compute the decision function of X.

# Arguments
- `model::AdaBoostClassifier`: The fitted AdaBoost model.
- `X::AbstractMatrix`: The input samples.

# Returns
- `Matrix{Float64}`: The decision function of the input samples.
"""
function decision_function(model::AdaBoostClassifier, X::AbstractMatrix)
    if !model.fitted
        throw(ErrorException("This AdaBoostClassifier instance is not fitted yet."))
    end

    n_samples = size(X, 1)
    scores = zeros(n_samples, model.n_classes_)

    for (estimator, weight) in zip(model.estimators_, model.estimator_weights_)
        if model.algorithm == :SAMME_R
            proba = estimator(X, type=:probs)
            current_score = log.(proba) .- log(1.0 / model.n_classes_)
        else
            predictions = estimator(X)
            current_score = zeros(n_samples, model.n_classes_)
            for (i, pred) in enumerate(predictions)
                current_score[i, findfirst(==(pred), model.classes_)] = 1
            end
        end
        scores .+= weight .* current_score
    end

    return scores
end

"""
    staged_predict(model::AdaBoostClassifier, X::AbstractMatrix)
Return a generator of predictions for each boosting iteration.

# Arguments
- `model::AdaBoostClassifier`: The fitted AdaBoost model.
- `X::AbstractMatrix`: The input samples.

# Returns
- `Channel`: A generator of predictions at each stage.
"""
function staged_predict(model::AdaBoostClassifier, X::AbstractMatrix)
    if !model.fitted
        throw(ErrorException("This AdaBoostClassifier instance is not fitted yet."))
    end

    n_samples = size(X, 1)
    scores = zeros(n_samples, model.n_classes_)

    return Channel() do channel
        for (estimator, weight) in zip(model.estimators_, model.estimator_weights_)
            if model.algorithm == :SAMME_R
                proba = estimator(X, type=:probs)
                current_score = log.(proba) .- log(1.0 / model.n_classes_)
            else
                predictions = estimator(X)
                current_score = zeros(n_samples, model.n_classes_)
                for (i, pred) in enumerate(predictions)
                    current_score[i, findfirst(==(pred), model.classes_)] = 1
                end
            end
            scores .+= weight .* current_score
            put!(channel, [model.classes_[argmax(scores[i, :])] for i in 1:n_samples])
        end
    end
end

"""
    staged_predict_proba(model::AdaBoostClassifier, X::AbstractMatrix)
Return a generator of predicted probabilities for each boosting iteration.

# Arguments
- `model::AdaBoostClassifier`: The fitted AdaBoost model.
- `X::AbstractMatrix`: The input samples.

# Returns
- `Channel`: A generator of predicted probabilities at each stage.
"""
function staged_predict_proba(model::AdaBoostClassifier, X::AbstractMatrix)
    if !model.fitted
        throw(ErrorException("This AdaBoostClassifier instance is not fitted yet."))
    end

    n_samples = size(X, 1)
    scores = zeros(n_samples, model.n_classes_)

    return Channel() do channel
        for (estimator, weight) in zip(model.estimators_, model.estimator_weights_)
            if model.algorithm == :SAMME_R
                proba = estimator(X, type=:probs)
                current_score = log.(proba) .- log(1.0 / model.n_classes_)
            else
                predictions = estimator(X)
                current_score = zeros(n_samples, model.n_classes_)
                for (i, pred) in enumerate(predictions)
                    current_score[i, findfirst(==(pred), model.classes_)] = 1
                end
            end
            scores .+= weight .* current_score
            proba = exp.(scores)
            proba ./= sum(proba, dims=2)
            put!(channel, proba)
        end
    end
end