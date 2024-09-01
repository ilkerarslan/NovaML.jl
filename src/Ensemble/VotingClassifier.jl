using Statistics
import ...NovaML: AbstractModel

using Statistics
import ...NovaML: AbstractModel

"""
    VotingClassifier <: AbstractModel

A Voting Classifier for combining multiple machine learning classifiers.

This classifier combines a number of estimators to create a single classifier that makes predictions
based on either hard voting (majority vote) or soft voting (weighted average of predicted probabilities).

# Fields
- `estimators::Vector{Tuple{String, Any}}`: List of (name, estimator) tuples.
- `voting::Symbol`: The voting strategy, either :hard for majority voting or :soft for probability voting.
- `weights::Union{Vector{Float64}, Nothing}`: Sequence of weights for each estimator in soft voting.
- `flatten_transform::Bool`: Affects the shape of transform output.
- `verbose::Bool`: If true, prints progress messages during fitting.

# Fitted Attributes
- `estimators_::Vector{Any}`: The fitted estimators.
- `classes_::Vector{Any}`: The class labels.
- `fitted::Bool`: Whether the classifier is fitted.

# Example
```julia
estimators = [("lr", LogisticRegression()), ("rf", RandomForestClassifier())]
vc = VotingClassifier(estimators=estimators, voting=:soft)
vc(X, y)  # Fit the classifier
predictions = vc(X_test)  # Make predictions
```
"""
mutable struct VotingClassifier <: AbstractModel
    estimators::Vector{Tuple{String, Any}}
    voting::Symbol
    weights::Union{Vector{Float64}, Nothing}
    flatten_transform::Bool
    verbose::Bool
    
    # Fitted attributes
    estimators_::Vector{Any}
    classes_::Vector{Any}
    fitted::Bool

    function VotingClassifier(;
        estimators::Vector{Tuple{String, Any}},
        voting::Symbol = :hard,
        weights::Union{Vector{Float64}, Nothing} = nothing,
        flatten_transform::Bool = true,
        verbose::Bool = false
    )
        @assert voting in [:hard, :soft] "voting must be either :hard or :soft"
        if weights !== nothing
            @assert length(weights) == length(estimators) "Number of weights must match number of estimators"
        end
        
        new(estimators, voting, weights, flatten_transform, verbose, 
            Any[], [], false)
    end
end

"""
    (vc::VotingClassifier)(X::AbstractMatrix, y::AbstractVector)
Fit the voting classifier.

# Arguments
- `X::AbstractMatrix`: The input samples.
- `y::AbstractVector`: The target values (class labels).

# Returns
- `VotingClassifier`: The fitted classifier.
"""
function (vc::VotingClassifier)(X::AbstractMatrix, y::AbstractVector)
    vc.classes_ = sort(unique(y))
    vc.estimators_ = Any[]
    
    for (name, estimator) in vc.estimators
        if vc.verbose
            println("Fitting $name...")
        end
        fitted_estimator = deepcopy(estimator)
        fitted_estimator(X, y)
        push!(vc.estimators_, fitted_estimator)
    end
    
    vc.fitted = true
    return vc
end

"""
    (vc::VotingClassifier)(X::AbstractMatrix; type=nothing)
Predict class labels for X.

# Arguments
- `X::AbstractMatrix`: The input samples.
- `type`: If set to :probs, return probability estimates for each class.

# Returns
- If `type` is `:probs`, returns probabilities of each class.
- Otherwise, returns predicted class labels.
"""
function (vc::VotingClassifier)(X::AbstractMatrix; type=nothing)
    if !vc.fitted
        throw(ErrorException("This VotingClassifier instance is not fitted yet. Call the model with training data before using it for predictions."))
    end
    
    n_samples = size(X, 1)
    n_classes = length(vc.classes_)
    n_estimators = length(vc.estimators_)
    
    if vc.voting == :soft
        proba = zeros(n_samples, n_classes)
        
        for (i, estimator) in enumerate(vc.estimators_)
            estimator_proba = estimator(X, type=:probs)
            
            
            if size(estimator_proba, 2) != n_classes
                temp_proba = zeros(n_samples, n_classes)
                for j in 1:n_samples
                    class_index = findfirst(c -> c == argmax(estimator_proba[j, :]), vc.classes_)
                    temp_proba[j, class_index] = 1.0
                end
                estimator_proba = temp_proba
            end
            
            weight = vc.weights !== nothing ? vc.weights[i] : 1.0
            proba .+= weight * estimator_proba
        end
        
        if type == :probs
            return proba ./ sum(vc.weights !== nothing ? vc.weights : ones(n_estimators))
        else
            return [vc.classes_[argmax(proba[i, :])] for i in 1:n_samples]
        end
    else  # hard voting
        predictions = Matrix{Any}(undef, n_samples, n_estimators)
        
        for (i, estimator) in enumerate(vc.estimators_)
            predictions[:, i] = estimator(X)
        end
        
        if type == :probs
            vote_counts = zeros(n_samples, n_classes)
            for i in 1:n_samples
                for j in 1:n_estimators
                    class_index = findfirst(c -> c == predictions[i, j], vc.classes_)
                    weight = vc.weights !== nothing ? vc.weights[j] : 1.0
                    vote_counts[i, class_index] += weight
                end
            end
            return vote_counts ./ sum(vc.weights !== nothing ? vc.weights : ones(n_estimators))
        else
            return [vc.classes_[argmax([count(==(c), row) for c in vc.classes_])] for row in eachrow(predictions)]
        end
    end
end

"""
    transform(vc::VotingClassifier, X::AbstractMatrix)
Return class labels or probabilities for X for each estimator.

# Arguments
- `vc::VotingClassifier`: The fitted voting classifier.
- `X::AbstractMatrix`: The input samples.

# Returns
- If `voting` is `:soft`, returns the probabilities for each class for each estimator.
- If `voting` is `:hard`, returns the class label predictions of each estimator.

The shape of the return depends on the `flatten_transform` parameter.
"""
function transform(vc::VotingClassifier, X::AbstractMatrix)
    if !vc.fitted
        throw(ErrorException("This VotingClassifier instance is not fitted yet. Call the model with training data before using transform."))
    end
    
    n_samples = size(X, 1)
    n_classes = length(vc.classes_)
    n_estimators = length(vc.estimators_)
    
    if vc.voting == :soft
        probas = [estimator(X, type=:probs) for estimator in vc.estimators_]
        
        if vc.flatten_transform
            return hcat(probas...)
        else
            return cat(probas..., dims=3)
        end
    else  # hard voting
        predictions = Matrix{Any}(undef, n_samples, n_estimators)
        
        for (i, estimator) in enumerate(vc.estimators_)
            predictions[:, i] = estimator(X)
        end
        
        return predictions
    end
end

"""
    Base.show(io::IO, vc::VotingClassifier)
Custom show method for VotingClassifier.

# Arguments
- `io::IO`: The I/O stream.
- `vc::VotingClassifier`: The voting classifier to display.
"""
function Base.show(io::IO, vc::VotingClassifier)
    estimator_names = join([name for (name, _) in vc.estimators], ", ")
    print(io, "VotingClassifier(estimators=[$estimator_names], ",
        "voting=$(vc.voting), ",
        "weights=$(vc.weights), ",
        "flatten_transform=$(vc.flatten_transform), ",
        "verbose=$(vc.verbose), ",
        "fitted=$(vc.fitted))")
end