using LinearAlgebra
using Statistics

import ...NovaML: AbstractModel

mutable struct LinearRegression <: AbstractModel
    w::Vector{Float64}
    b::Float64
    fit_intercept::Bool
    normalize::Bool
    copy_X::Bool
    n_jobs::Union{Int, Nothing}
    positive::Bool
    fitted::Bool

    function LinearRegression(;
        fit_intercept::Bool=true,
        normalize::Bool=false,
        copy_X::Bool=true,
        n_jobs::Union{Int, Nothing}=nothing,
        positive::Bool=false
    )
        new(
            Vector{Float64}(),
            0.0,
            fit_intercept,
            normalize,
            copy_X,
            n_jobs,
            positive,
            false
        )
    end
end

function (model::LinearRegression)(X::AbstractMatrix{T}, y::AbstractVector{T}) where T <: Real
    n_samples, n_features = size(X)
    
    if model.copy_X
        X = copy(X)
    end
    
    if model.normalize
        X_mean = mean(X, dims=1)
        X_std = std(X, dims=1)
        X = (X .- X_mean) ./ X_std
    end
    
    if model.fit_intercept
        X = hcat(ones(n_samples), X)
    end
    
    if model.positive
        # Use non-negative least squares
        model.w = nnls(X, y)
    else
        # Use ordinary least squares
        model.w = X \ y
    end
    
    if model.fit_intercept
        model.b = model.w[1]
        model.w = model.w[2:end]
    else
        model.b = 0.0
    end
    
    model.fitted = true
    return model
end

function (model::LinearRegression)(X::AbstractMatrix{T}) where T <: Real
    if !model.fitted
        throw(ErrorException("This LinearRegression instance is not fitted yet. Call the model with training data before using it for predictions."))
    end
    
    if model.normalize
        X_mean = mean(X, dims=1)
        X_std = std(X, dims=1)
        X = (X .- X_mean) ./ X_std
    end
    
    return X * model.w .+ model.b
end

function nnls(X::AbstractMatrix{T}, y::AbstractVector{T}) where T <: Real
    n_features = size(X, 2)
    w = zeros(n_features)
    residual = y - X * w
    
    for _ in 1:1000  # Max iterations
        gradient = X' * residual
        if all(w .> 0) && all(gradient .<= 0)
            break
        end
        
        index = argmax(gradient)
        w[index] += gradient[index]
        w = max.(w, 0)
        residual = y - X * w
    end
    
    return w
end

function Base.show(io::IO, model::LinearRegression)
    println(io, "LinearRegression(")
    println(io, "  fit_intercept=$(model.fit_intercept),")
    println(io, "  normalize=$(model.normalize),")
    println(io, "  copy_X=$(model.copy_X),")
    println(io, "  n_jobs=$(model.n_jobs),")
    println(io, "  positive=$(model.positive),")
    println(io, "  fitted=$(model.fitted)")
    print(io, ")")
end