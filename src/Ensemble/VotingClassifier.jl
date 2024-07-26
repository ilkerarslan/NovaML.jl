using Statistics
import ...NovaML: AbstractModel

using Statistics
import ...NovaML: AbstractModel

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

function Base.show(io::IO, vc::VotingClassifier)
    estimator_names = join([name for (name, _) in vc.estimators], ", ")
    print(io, "VotingClassifier(estimators=[$estimator_names], ",
        "voting=$(vc.voting), ",
        "weights=$(vc.weights), ",
        "flatten_transform=$(vc.flatten_transform), ",
        "verbose=$(vc.verbose), ",
        "fitted=$(vc.fitted))")
end