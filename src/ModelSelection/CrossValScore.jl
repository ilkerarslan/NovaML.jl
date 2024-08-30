using Statistics
using Random
import StatsBase

import ...NovaML: default_score

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

    # Determine the number of threads to use
    n_threads = if n_jobs === nothing
        Threads.nthreads()
    elseif n_jobs == -1
        Threads.nthreads()
    else
        min(n_jobs, Threads.nthreads())
    end

    scores = Vector{Float64}(undef, length(indices))
    
    # Use threaded @sync for parallel execution
    @sync begin
        for (fold, test_idx) in collect(enumerate(indices))
            Threads.@spawn begin
                train_idx = setdiff(1:n_samples, test_idx)
                X_train, X_test = X[train_idx, :], X[test_idx, :]
                y_train, y_test = y[train_idx], y[test_idx]

                try
                    # Create a deep copy of the estimator for each thread
                    local_estimator = deepcopy(estimator)

                    # Train the model
                    local_estimator(X_train, y_train)

                    # Make predictions
                    y_pred = local_estimator(X_test)

                    # Ensure y_pred is a vector
                    y_pred = vec(y_pred)

                    # Compute the score
                    score = if scoring === nothing
                        default_score(local_estimator, X_test, y_test, y_pred)
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

                    scores[fold] = score
                catch e
                    @error "Error during cross-validation in fold $fold" exception=(e, catch_backtrace())
                    rethrow(e)
                end
            end
        end
    end

    return scores
end
