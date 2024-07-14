# File: src/ModelSelection/ValidationCurve.jl

using Statistics
using Random

import ...NovaML: default_score

function validation_curve(
    estimator,
    X::AbstractMatrix,
    y::AbstractVector;
    param_dict::Dict{Symbol, Symbol},
    param_range::AbstractVector,
    groups=nothing,
    cv=5,
    scoring=nothing,
    n_jobs=nothing,
    verbose=0
)
    n_samples, n_features = size(X)
    
    # Extract model type and parameter from the dictionary
    model_type, param_name = first(param_dict)
    
    # Create CV iterator
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
    
    # Prepare results containers
    n_params = length(param_range)
    train_scores = zeros(n_params, length(indices))
    test_scores = zeros(n_params, length(indices))
    
    # Perform cross-validation for each parameter value
    for (param_idx, param_value) in enumerate(param_range)
        for (fold, test_idx) in enumerate(indices)
            train_idx = setdiff(1:n_samples, test_idx)
            X_train, X_test = X[train_idx, :], X[test_idx, :]
            y_train, y_test = y[train_idx], y[test_idx]
            
            try
                local_estimator = deepcopy(estimator)
                train_score, test_score = fit_and_score(
                    local_estimator,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    param_value,
                    model_type,
                    param_name,
                    scoring
                )
                
                # Check for NaN and replace with 0 if necessary
                if isnan(train_score) || isnan(test_score)
                    @warn "NaN score detected in fold $fold for param value $param_value. Replacing with 0."
                    train_score = isnan(train_score) ? 0.0 : train_score
                    test_score = isnan(test_score) ? 0.0 : test_score
                end
                
                train_scores[param_idx, fold] = train_score
                test_scores[param_idx, fold] = test_score
            catch e
                @error "Error during validation in fold $fold for param value $param_value" exception=(e, catch_backtrace())
                rethrow(e)
            end
        end
    end
    
    return train_scores, test_scores
end

# Function to fit and score a single fold
function fit_and_score(estimator, X_train, y_train, X_test, y_test, param_value, model_type, param_name, scoring)
    # Update the parameter
    if hasproperty(estimator, :steps)  # Check if it's a Pipe-like object
        for (i, step) in enumerate(estimator.steps)
            if Symbol(typeof(step)) == model_type
                new_step = deepcopy(step)
                setproperty!(new_step, param_name, param_value)
                estimator.steps[i] = new_step
                break
            end
        end
    else
        if Symbol(typeof(estimator)) == model_type
            setproperty!(estimator, param_name, param_value)
        else
            error("Estimator type does not match the specified model type")
        end
    end
    
    # Fit the estimator
    estimator(X_train, y_train)
    
    # Score the estimator
    if scoring === nothing
        # Use default scoring method
        train_score = default_score(y_train, estimator(X_train))
        test_score = default_score(y_test, estimator(X_test))
    elseif scoring isa Function
        train_score = scoring(y_train, estimator(X_train))
        test_score = scoring(y_test, estimator(X_test))
    elseif scoring isa String
        if scoring == "accuracy"
            train_score = sum(y_train .== estimator(X_train)) / length(y_train)
            test_score = sum(y_test .== estimator(X_test)) / length(y_test)
        elseif scoring == "mse"
            train_score = mean((y_train .- estimator(X_train)).^2)
            test_score = mean((y_test .- estimator(X_test)).^2)
        else
            error("Unknown scoring method: $scoring")
        end
    else
        error("Invalid scoring parameter type. Expected nothing, Function, or String.")
    end
    
    return train_score, test_score
end
