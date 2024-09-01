using Random
using Statistics
import ...NovaML: AbstractModel

mutable struct BaggingClassifier <: AbstractModel
    base_estimator::AbstractModel
    n_estimators::Int
    max_samples::Union{Int, Float64}
    max_features::Union{Int, Float64}
    bootstrap::Bool
    bootstrap_features::Bool
    oob_score::Bool
    warm_start::Bool
    random_state::Union{Int, Nothing}
    verbose::Int

    # Fitted attributes
    estimators_::Vector{AbstractModel}
    estimators_features_::Vector{Vector{Int}}
    classes_::Vector
    n_classes_::Int
    oob_score_::Union{Float64, Nothing}
    oob_decision_function_::Union{Matrix{Float64}, Nothing}
    fitted::Bool

    function BaggingClassifier(;
        base_estimator=DecisionTreeClassifier(),
        n_estimators::Int=10,
        max_samples::Union{Int, Float64}=1.0,
        max_features::Union{Int, Float64}=1.0,
        bootstrap::Bool=true,
        bootstrap_features::Bool=false,
        oob_score::Bool=false,
        warm_start::Bool=false,
        random_state::Union{Int, Nothing}=nothing,
        verbose::Int=0
    )
        new(
            base_estimator, n_estimators, max_samples, max_features,
            bootstrap, bootstrap_features, oob_score, warm_start,
            random_state, verbose,
            AbstractModel[], Vector{Int}[], [], 0, nothing, nothing, false
        )
    end
end

function (bc::BaggingClassifier)(X::AbstractMatrix, y::AbstractVector)
    n_samples, n_features = size(X)
    bc.classes_ = sort(unique(y))
    bc.n_classes_ = length(bc.classes_)

    if bc.random_state !== nothing
        Random.seed!(bc.random_state)
    end

    if !bc.warm_start || isempty(bc.estimators_)
        bc.estimators_ = Vector{AbstractModel}(undef, bc.n_estimators)
        bc.estimators_features_ = Vector{Vector{Int}}(undef, bc.n_estimators)
    elseif length(bc.estimators_) < bc.n_estimators
        append!(bc.estimators_, Vector{AbstractModel}(undef, bc.n_estimators - length(bc.estimators_)))
        append!(bc.estimators_features_, Vector{Vector{Int}}(undef, bc.n_estimators - length(bc.estimators_features_)))
    elseif length(bc.estimators_) > bc.n_estimators
        bc.estimators_ = bc.estimators_[1:bc.n_estimators]
        bc.estimators_features_ = bc.estimators_features_[1:bc.n_estimators]
    end

    # Parallel training of estimators
    Threads.@threads for i in 1:bc.n_estimators
        if !isdefined(bc.estimators_, i) || bc.estimators_[i] === nothing
            estimator = deepcopy(bc.base_estimator)
            
            if bc.bootstrap
                indices = rand(1:n_samples, n_samples)
            else
                indices = 1:n_samples
            end

            if bc.bootstrap_features
                feature_indices = rand(1:n_features, n_features)
            else
                feature_indices = 1:n_features
            end

            X_train = X[indices, feature_indices]
            y_train = y[indices]

            estimator(X_train, y_train)
            bc.estimators_[i] = estimator
            bc.estimators_features_[i] = feature_indices
        end
    end

    if bc.oob_score
        bc._compute_oob_score(X, y)
    end

    bc.fitted = true
    return bc
end

function (bc::BaggingClassifier)(X::AbstractMatrix; type=nothing)
    if !bc.fitted
        throw(ErrorException("This BaggingClassifier instance is not fitted yet. Call the model with training data before using it for predictions."))
    end

    n_samples = size(X, 1)
    
    if type == :probs
        proba = zeros(n_samples, bc.n_classes_)
        Threads.@threads for i in 1:bc.n_estimators
            proba .+= bc.estimators_[i](X[:, bc.estimators_features_[i]], type=:probs)
        end
        proba ./= bc.n_estimators
        return proba
    else
        predictions = Matrix{Int}(undef, n_samples, bc.n_estimators)
        Threads.@threads for i in 1:bc.n_estimators
            predictions[:, i] = bc.estimators_[i](X[:, bc.estimators_features_[i]])
        end
        return [bc.classes_[argmax(count(==(c), row) for c in bc.classes_)] for row in eachrow(predictions)]
    end
end

function _compute_oob_score(bc::BaggingClassifier, X::AbstractMatrix, y::AbstractVector)
    n_samples = size(X, 1)
    n_classes = length(bc.classes_)
    
    oob_decision_function = zeros(n_samples, n_classes)
    n_oob_predictions = zeros(Int, n_samples)
    
    Threads.@threads for i in 1:length(bc.estimators_)
        estimator = bc.estimators_[i]
        features = bc.estimators_features_[i]
        unsampled_indices = setdiff(1:n_samples, _generate_indices(bc, n_samples))
        
        if !isempty(unsampled_indices)
            X_oob = X[unsampled_indices, features]
            y_pred_proba = estimator(X_oob, type=:probs)
            
            oob_decision_function[unsampled_indices, :] .+= y_pred_proba
            n_oob_predictions[unsampled_indices] .+= 1
        end
    end
    
    oob_decision_function ./= max.(1, n_oob_predictions)
    
    y_pred = [bc.classes_[argmax(probs)] for probs in eachrow(oob_decision_function)]
    bc.oob_score_ = mean(y_pred .== y)
    bc.oob_decision_function_ = oob_decision_function
end

function _generate_indices(bc::BaggingClassifier, n_samples::Int)
    if bc.max_samples isa Int
        n_samples = min(bc.max_samples, n_samples)
    else
        n_samples = round(Int, bc.max_samples * n_samples)
    end
    
    if bc.bootstrap
        return rand(1:n_samples, n_samples)
    else
        return shuffle(1:n_samples)[1:n_samples]
    end
end

function Base.show(io::IO, bc::BaggingClassifier)
    print(io, "BaggingClassifier(base_estimator=$(bc.base_estimator), ",
              "n_estimators=$(bc.n_estimators), ",
              "max_samples=$(bc.max_samples), ",
              "max_features=$(bc.max_features), ",
              "bootstrap=$(bc.bootstrap), ",
              "bootstrap_features=$(bc.bootstrap_features), ",
              "oob_score=$(bc.oob_score), ",
              "warm_start=$(bc.warm_start), ",
              "random_state=$(bc.random_state), ",
              "verbose=$(bc.verbose))")
end