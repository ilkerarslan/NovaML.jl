using Statistics
using Random

"""
    cross_val_score(estimator, X, y; cv=5, scoring=nothing, n_jobs=nothing)

Evaluate a score by cross-validation.

# Arguments
- `estimator`: The object to use to fit the data.
- `X`: The data to fit. Can be a matrix or any array-like structure.
- `y`: The target variable to try to predict.
- `cv`: Determines the cross-validation splitting strategy. Default is 5.
- `scoring`: A string or a callable to evaluate the predictions on the test set.
             If nothing, the estimator's default scorer is used.
- `n_jobs`: The number of jobs to run in parallel. `nothing` means 1.

# Returns
- An array of scores of the estimator for each run of the cross validation.
"""
function cross_val_score(estimator, X, y; cv=5, scoring=nothing, n_jobs=nothing)
    n_samples = size(X, 1)
    
    # Create CV iterator
    if typeof(cv) <: Integer
        indices = collect(Iterators.partition(Random.shuffle(1:n_samples), n_samples ÷ cv))
    else
        # Assume cv is an iterable of (train, test) indices
        indices = cv
    end

    scores = Float64[]

    for (test_idx, train_idx) in zip(indices, setdiff.(Ref(1:n_samples), indices))
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train the model
        model = estimator(X_train, y_train)

        # Make predictions
        y_pred = model(X_test)

        # Compute the score
        if scoring === nothing
            # Use default scoring method if available
            if hasmethod(score, Tuple{typeof(estimator), typeof(X_test), typeof(y_test)})
                push!(scores, score(estimator, X_test, y_test))
            else
                error("No scoring method provided and the estimator doesn't have a default scoring method.")
            end
        elseif typeof(scoring) <: Function
            push!(scores, scoring(y_test, y_pred))
        elseif typeof(scoring) <: String
            # Implement common scoring methods
            if scoring == "accuracy"
                push!(scores, sum(y_test .== y_pred) / length(y_test))
            elseif scoring == "mse"
                push!(scores, mean((y_test .- y_pred).^2))
            else
                error("Unknown scoring method: $scoring")
            end
        else
            error("Invalid scoring parameter type. Expected nothing, Function, or String.")
        end
    end

    return scores
end

# Helper function to compute default score if the estimator supports it
function score(estimator, X, y)
    if hasmethod(score, Tuple{typeof(estimator), typeof(X), typeof(y)})
        return score(estimator, X, y)
    else
        error("The estimator does not have a default scoring method.")
    end
end