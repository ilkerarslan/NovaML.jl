using LinearAlgebra
using Random
using Optim

import ...NovaML: AbstractModel

mutable struct Lasso <: AbstractModel
    w::Vector{Float64}
    b::Float64
    α::Float64
    fit_intercept::Bool
    max_iter::Int
    tol::Float64
    random_state::Union{Int, Nothing}
    selection::String
    fitted::Bool

    function Lasso(;
        α::Float64 = 1.0,
        fit_intercept::Bool = true,
        max_iter::Int = 1000,
        tol::Float64 = 1e-4,
        random_state::Union{Int, Nothing} = nothing,
        selection::String = "cyclic"
    )
        if α < 0
            throw(ArgumentError("α must be non-negative"))
        end
        if !(selection in ["cyclic", "random"])
            throw(ArgumentError("selection must be 'cyclic' or 'random'"))
        end
        new(
            Vector{Float64}(),
            0.0,
            α,
            fit_intercept,
            max_iter,
            tol,
            random_state,
            selection,
            false
        )
    end
end

function (model::Lasso)(X::AbstractMatrix{T}, y::AbstractVector{T}) where T <: Real
    n_samples, n_features = size(X)
    
    if model.random_state !== nothing
        Random.seed!(model.random_state)
    end
    
    if model.fit_intercept
        X = hcat(ones(n_samples), X)
        n_features += 1
    end
    
    # Initialize weights
    w = zeros(n_features)
    
    # Objective function
    function f(w)
        return 0.5 * sum((y - X * w).^2) / n_samples + model.α * sum(abs.(w[2:end]))
    end
    
    # Gradient function
    function g!(G, w)
        G .= -X' * (y - X * w) / n_samples
        G[2:end] .+= model.α * sign.(w[2:end])
    end
    
    # Optimization
    res = optimize(f, g!, w, LBFGS(), Optim.Options(iterations=model.max_iter, g_tol=model.tol))
    
    # Extract results
    if model.fit_intercept
        model.b = Optim.minimizer(res)[1]
        model.w = Optim.minimizer(res)[2:end]
    else
        model.w = Optim.minimizer(res)
        model.b = 0.0
    end
    
    model.fitted = true
    return model
end

function (model::Lasso)(X::AbstractMatrix{T}) where T <: Real
    if !model.fitted
        throw(ErrorException("This Lasso instance is not fitted yet. Call the model with training data before using it for predictions."))
    end
    
    return X * model.w .+ model.b
end

function Base.show(io::IO, model::Lasso)
    println(io, "Lasso(")
    println(io, "  α=$(model.α),")
    println(io, "  fit_intercept=$(model.fit_intercept),")
    println(io, "  max_iter=$(model.max_iter),")
    println(io, "  tol=$(model.tol),")
    println(io, "  random_state=$(model.random_state),")
    println(io, "  selection=\"$(model.selection)\",")
    println(io, "  fitted=$(model.fitted)")
    print(io, ")")
end