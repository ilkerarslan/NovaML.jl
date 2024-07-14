using Statistics
using Random
import StatsBase

export cross_val_score

function cross_val_score(estimator, X, y; cv=5, scoring=nothing, n_jobs=nothing)
    n_samples = size(X, 1)
    
    if cv isa Integer
        # Create cv folds
        fold_sizes = fill(n_samples ÷ cv, cv)
        fold_sizes[1:n_samples % cv] .+= 1
        indices = Vector{Int}[]
        start = 1
        for size in fold_sizes
            push!(indices, start:(start+size-1))
            start += size
        end
    else
        indices = cv
    end

    scores = Float64[]

    for (fold, test_idx) in enumerate(indices)
        train_idx = setdiff(1:n_samples, test_idx)
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        try
            # Train the model
            estimator(X_train, y_train)

            # Make predictions
            y_pred = estimator(X_test)

            # Ensure y_pred is a vector
            y_pred = vec(y_pred)

            # Compute the score
            score = if scoring === nothing
                default_score(estimator, X_test, y_test, y_pred)
            elseif scoring isa Function
                scoring(y_test, y_pred)
            elseif scoring isa String
                if scoring == "accuracy"
                    sum(y_test .== y_pred) / length(y_test)
                elseif scoring == "mse"
                    mean((y_test .- y_pred).^2)
                else
                    error("Unknown scoring method: $scoring")
                end
            else
                error("Invalid scoring parameter type. Expected nothing, Function, or String.")
            end

            # Check for NaN and replace with 0 if necessary
            if isnan(score)
                @warn "NaN score detected in fold $fold. Replacing with 0."
                score = 0.0
            end

            push!(scores, score)
        catch e
            @error "Error during cross-validation in fold $fold" exception=(e, catch_backtrace())
            rethrow(e)
        end
    end

    return scores
end

# Default scoring function
function default_score(estimator, X, y, y_pred)
    if y isa AbstractVector{<:Number} && y_pred isa AbstractVector{<:Number}
        # R-squared for regression
        ss_res = sum((y .- y_pred).^2)
        ss_tot = sum((y .- mean(y)).^2)
        return max(0, 1 - ss_res / ss_tot)  # Ensure non-negative R-squared
    else
        # Accuracy for classification
        return sum(y .== y_pred) / length(y)
    end
end