using LinearAlgebra
using Random
using Optim
using SparseArrays

import ...NovaML: AbstractModel

mutable struct ElasticNet <: AbstractModel
    # Model parameters
    w::Vector{Float64}  # Coefficients
    b::Float64          # Intercept
    
    # Hyperparameters
    α::Float64
    l1_ratio::Float64
    fit_intercept::Bool
    max_iter::Int
    tol::Float64
    positive::Bool
    random_state::Union{Int, Nothing}
    selection::Symbol

    # Attributes
    n_iter_::Union{Int, Nothing}
    dual_gap_::Union{Float64, Nothing}
    n_features_in_::Union{Int, Nothing}
    feature_names_in_::Union{Vector{String}, Nothing}
    sparse_coef_::Union{SparseVector{Float64, Int}, Nothing}
    
    # State
    fitted::Bool

    function ElasticNet(;
        α::Float64 = 1.0,
        l1_ratio::Float64 = 0.5,
        fit_intercept::Bool = true,
        max_iter::Int = 1000,
        tol::Float64 = 1e-4,
        positive::Bool = false,
        random_state::Union{Int, Nothing} = nothing,
        selection::Symbol = :cyclic
    )
        @assert 0 <= l1_ratio <= 1 "l1_ratio must be between 0 and 1"
        @assert α >= 0 "α must be non-negative"
        @assert selection in [:cyclic, :random] "selection must be :cyclic or :random"
        
        new(
            Vector{Float64}(), 0.0,  # w, b
            α, l1_ratio, fit_intercept, max_iter, tol, positive, random_state, selection,
            nothing, nothing, nothing, nothing, nothing,  # n_iter_, dual_gap_, n_features_in_, feature_names_in_, sparse_coef_
            false  # fitted
        )
    end
end

function (model::ElasticNet)(X::AbstractMatrix{T}, y::AbstractVector{T}) where T <: Real
    n_samples, n_features = size(X)
    model.n_features_in_ = n_features
    
    if model.random_state !== nothing
        Random.seed!(model.random_state)
    end
    
    # Center the data if fit_intercept is true
    X_centered = model.fit_intercept ? X .- mean(X, dims=1) : X
    y_centered = model.fit_intercept ? y .- mean(y) : y
    
    # Initialize coefficients
    model.w = zeros(n_features)
    
    # Optimization function
    function f(β)
        Xβ = X_centered * β
        mse = sum((y_centered - Xβ).^2) / (2 * n_samples)
        l1 = model.α * model.l1_ratio * sum(abs.(β))
        l2 = 0.5 * model.α * (1 - model.l1_ratio) * sum(β.^2)
        return mse + l1 + l2
    end
    
    # Gradient function
    function g!(G, β)
        Xβ = X_centered * β
        G .= -X_centered' * (y_centered - Xβ) / n_samples .+
             model.α * model.l1_ratio * sign.(β) .+
             model.α * (1 - model.l1_ratio) * β
    end
    
    # Optimize
    result = optimize(f, g!, model.w, LBFGS(), Optim.Options(iterations=model.max_iter, g_tol=model.tol))
    
    # Update model parameters
    model.w = Optim.minimizer(result)
    if model.fit_intercept
        model.b = mean(y) - dot(mean(X, dims=1), model.w)
    else
        model.b = 0.0
    end
    
    # Update other attributes
    model.n_iter_ = Optim.iterations(result)
    model.dual_gap_ = Optim.minimum(result)
    model.sparse_coef_ = sparse(model.w)
    model.fitted = true
    
    return model
end

function (model::ElasticNet)(X::AbstractMatrix{T}; type=nothing) where T <: Real
    if !model.fitted
        throw(ErrorException("This ElasticNet instance is not fitted yet. Call the model with training data before using it for predictions."))
    end
    
    if type == :coef
        return model.w
    elseif type == :intercept
        return model.b
    else
        return X * model.w .+ model.b
    end
end

# Helper functions
function path(X::AbstractMatrix{T}, y::AbstractVector{T}; 
              l1_ratio::Float64=0.5, eps::Float64=1e-3, n_alphas::Int=100, 
              alphas::Union{Nothing, Vector{Float64}}=nothing) where T <: Real
    
    n_samples, n_features = size(X)
    
    if alphas === nothing
        alpha_max = maximum(abs.(X' * y)) / (n_samples * l1_ratio)
        alphas = exp.(range(log(alpha_max), log(eps * alpha_max), length=n_alphas))
    end
    
    coefs = zeros(n_features, length(alphas))
    
    for (i, alpha) in enumerate(alphas)
        model = ElasticNet(α=alpha, l1_ratio=l1_ratio)
        model(X, y)
        coefs[:, i] = model.w
    end
    
    return alphas, coefs
end

# Implement other necessary methods (get_params, set_params, etc.) as needed

function Base.show(io::IO, model::ElasticNet)
    println(io, "ElasticNet(")
    println(io, "  α=$(model.α),")
    println(io, "  l1_ratio=$(model.l1_ratio),")
    println(io, "  fit_intercept=$(model.fit_intercept),")
    println(io, "  max_iter=$(model.max_iter),")
    println(io, "  tol=$(model.tol),")
    println(io, "  positive=$(model.positive),")
    println(io, "  random_state=$(model.random_state),")
    println(io, "  selection=$(model.selection),")
    println(io, "  fitted=$(model.fitted)")
    print(io, ")")
end