using Random
using Statistics
using Distributions
using LinearAlgebra

import ...NovaML: AbstractModel

export RANSACRegressor

mutable struct RANSACRegressor <: AbstractModel
    estimator::Union{Nothing, AbstractModel}
    min_samples::Union{Int, Float64}
    residual_threshold::Union{Nothing, Float64}
    is_data_valid::Union{Nothing, Function}
    is_model_valid::Union{Nothing, Function}
    max_trials::Int
    max_skips::Float64
    stop_n_inliers::Float64
    stop_score::Float64
    stop_probability::Float64
    loss::Union{String, Function}
    random_state::Union{Nothing, Int}

    # Fitted attributes
    estimator_::Union{Nothing, AbstractModel}
    n_trials_::Int
    inlier_mask_::Vector{Bool}
    n_skips_no_inliers_::Int
    n_skips_invalid_data_::Int
    n_skips_invalid_model_::Int
    n_features_in_::Int
    feature_names_in_::Vector{String}

    function RANSACRegressor(;
        estimator=nothing,
        min_samples=nothing,
        residual_threshold=nothing,
        is_data_valid=nothing,
        is_model_valid=nothing,
        max_trials=100,
        max_skips=Inf,
        stop_n_inliers=Inf,
        stop_score=Inf,
        stop_probability=0.99,
        loss="absolute_error",
        random_state=nothing
    )
        new(estimator, min_samples, residual_threshold, is_data_valid, is_model_valid,
            max_trials, max_skips, stop_n_inliers, stop_score, stop_probability,
            loss, random_state,
            nothing, 0, Bool[], 0, 0, 0, 0, String[])
    end
end

function (ransac::RANSACRegressor)(X::AbstractMatrix, y::AbstractVector; sample_weight=nothing)
    n_samples, n_features = size(X)
    
    if ransac.estimator === nothing
        ransac.estimator = LinearRegression()
    end
    
    if isa(ransac.min_samples, Float64) && 0 < ransac.min_samples < 1
        ransac.min_samples = max(ceil(Int, ransac.min_samples * n_samples), 2)
    elseif isa(ransac.min_samples, Int) && ransac.min_samples > 0
        ransac.min_samples = min(ransac.min_samples, n_samples)
    elseif ransac.min_samples === nothing
        ransac.min_samples = n_features + 1
    else
        throw(ArgumentError("min_samples must be a positive integer or a float in (0, 1)"))
    end
    
    if ransac.residual_threshold === nothing
        y_median = median(y)
        ransac.residual_threshold = median(abs.(y .- y_median))
    end    
    
    if ransac.random_state !== nothing
        Random.seed!(ransac.random_state)
    end
    
    ransac.n_features_in_ = n_features
    ransac.feature_names_in_ = ["feature_$i" for i in 1:n_features]
    
    best_score = -Inf
    best_inlier_mask = falses(n_samples)
    best_inlier_X = Matrix{Float64}(undef, 0, 0)
    best_inlier_y = Vector{Float64}(undef, 0)
    
    n_inliers_best = 0
    
    ransac.n_trials_ = 0
    ransac.n_skips_no_inliers_ = 0
    ransac.n_skips_invalid_data_ = 0
    ransac.n_skips_invalid_model_ = 0
    
    while ransac.n_trials_ < ransac.max_trials
        ransac.n_trials_ += 1
        
        sample_idxs = sample(1:n_samples, ransac.min_samples, replace=false)
        X_subset = X[sample_idxs, :]
        y_subset = y[sample_idxs]
        
        if ransac.is_data_valid !== nothing && !ransac.is_data_valid(X_subset, y_subset)
            ransac.n_skips_invalid_data_ += 1
            if ransac.n_skips_invalid_data_ > ransac.max_skips
                break
            end
            continue
        end
        
        sample_model = deepcopy(ransac.estimator)
        sample_model(X_subset, y_subset)
        
        if ransac.is_model_valid !== nothing && !ransac.is_model_valid(sample_model, X_subset, y_subset)
            ransac.n_skips_invalid_model_ += 1
            if ransac.n_skips_invalid_model_ > ransac.max_skips
                break
            end
            continue
        end
        
        residuals = _calculate_residuals(ransac, sample_model, X, y)
        inlier_mask = abs.(residuals) .<= ransac.residual_threshold
        n_inliers = sum(inlier_mask)
        
        if n_inliers == 0
            ransac.n_skips_no_inliers_ += 1
            if ransac.n_skips_no_inliers_ > ransac.max_skips
                break
            end
            continue
        end
        
        inlier_X = X[inlier_mask, :]
        inlier_y = y[inlier_mask]
        
        inlier_model = deepcopy(ransac.estimator)
        inlier_model(inlier_X, inlier_y)
        
        score = _score_model(inlier_model, X, y)
        
        if score > best_score
            best_score = score
            best_inlier_mask = inlier_mask
            best_inlier_X = inlier_X
            best_inlier_y = inlier_y
            n_inliers_best = n_inliers
        end
        
        if n_inliers_best >= ransac.stop_n_inliers || best_score >= ransac.stop_score
            break
        end
        
        if n_inliers > 0            
            dynamic_max_trials = _dynamic_max_trials(n_inliers, n_samples, ransac.min_samples, ransac.stop_probability)
            
            if ransac.n_trials_ >= dynamic_max_trials
                break
            end
        end
    end
    
    if n_inliers_best == 0
        throw(ErrorException("RANSAC could not find a valid consensus set"))
    end
    
    ransac.estimator_ = deepcopy(ransac.estimator)
    ransac.estimator_(best_inlier_X, best_inlier_y)
    ransac.inlier_mask_ = best_inlier_mask
    
    return ransac
end

function (ransac::RANSACRegressor)(X::AbstractMatrix)
    if ransac.estimator_ === nothing
        throw(ErrorException("This RANSACRegressor instance is not fitted yet. Call with training data before using it for predictions."))
    end
    return ransac.estimator_(X)
end

function _calculate_residuals(ransac::RANSACRegressor, model::AbstractModel, X::AbstractMatrix, y::AbstractVector)
    y_pred = model(X)
    if ransac.loss == "absolute_error"
        return abs.(y .- y_pred)
    elseif ransac.loss == "squared_error"
        return (y .- y_pred).^2
    elseif isa(ransac.loss, Function)
        return ransac.loss(y, y_pred)
    else
        throw(ArgumentError("Invalid loss. Use 'absolute_error', 'squared_error', or a callable."))
    end
end

function _score_model(model::AbstractModel, X::AbstractMatrix, y::AbstractVector)
    if hasmethod(score, (typeof(model), AbstractMatrix, AbstractVector))
        return score(model, X, y)
    else
        y_pred = model(X)
        return 1 - sum((y .- y_pred).^2) / sum((y .- mean(y)).^2)
    end
end

function _dynamic_max_trials(n_inliers::Int, n_samples::Int, min_samples::Int, prob::Float64)
    inlier_ratio = n_inliers / n_samples

    if inlier_ratio ≈ 0 || inlier_ratio^min_samples ≈ 1
        return typemax(Int)
    end

    nom = log(1 - prob)
    denom = log(1 - inlier_ratio^min_samples)   

    if isfinite(nom) && isfinite(denom) && denom != 0
        result = ceil(Int, nom / denom)
        return result
    else
        return typemax(Int)
    end
end

# Helper methods

function get_params(ransac::RANSACRegressor; deep::Bool=true)
    params = Dict{Symbol, Any}(
        :estimator => ransac.estimator,
        :min_samples => ransac.min_samples,
        :residual_threshold => ransac.residual_threshold,
        :is_data_valid => ransac.is_data_valid,
        :is_model_valid => ransac.is_model_valid,
        :max_trials => ransac.max_trials,
        :max_skips => ransac.max_skips,
        :stop_n_inliers => ransac.stop_n_inliers,
        :stop_score => ransac.stop_score,
        :stop_probability => ransac.stop_probability,
        :loss => ransac.loss,
        :random_state => ransac.random_state
    )
    if deep && ransac.estimator !== nothing
        estimator_params = get_params(ransac.estimator, deep=true)
        for (key, value) in estimator_params
            params[Symbol("estimator__$key")] = value
        end
    end
    return params
end

function set_params!(ransac::RANSACRegressor; kwargs...)
    for (key, value) in kwargs
        if startswith(string(key), "estimator__")
            if ransac.estimator === nothing
                throw(ArgumentError("Estimator is not set. Cannot set its parameters."))
            end
            param = Symbol(split(string(key), "__")[2])
            set_params!(ransac.estimator; Dict(param => value)...)
        else
            setproperty!(ransac, key, value)
        end
    end
    return ransac
end

function score(ransac::RANSACRegressor, X::AbstractMatrix, y::AbstractVector)
    if ransac.estimator_ === nothing
        throw(ErrorException("This RANSACRegressor instance is not fitted yet. Call with training data before using it for scoring."))
    end
    return _score_model(ransac.estimator_, X, y)
end

function Base.show(io::IO, ransac::RANSACRegressor)
    println(io, "RANSACRegressor(")
    println(io, "  estimator=$(ransac.estimator),")
    println(io, "  min_samples=$(ransac.min_samples),")
    println(io, "  residual_threshold=$(ransac.residual_threshold),")
    println(io, "  max_trials=$(ransac.max_trials),")
    println(io, "  max_skips=$(ransac.max_skips),")
    println(io, "  stop_n_inliers=$(ransac.stop_n_inliers),")
    println(io, "  stop_score=$(ransac.stop_score),")
    println(io, "  stop_probability=$(ransac.stop_probability),")
    println(io, "  loss=$(ransac.loss),")
    println(io, "  random_state=$(ransac.random_state),")
    println(io, "  fitted=$(ransac.estimator_ !== nothing)")
    print(io, ")")
end