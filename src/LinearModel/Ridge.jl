using LinearAlgebra
using SparseArrays
using Random
using Optim

import ...NovaML: AbstractModel

mutable struct Ridge <: AbstractModel
    w::Vector{Float64}
    b::Float64
    α::Float64
    fit_intercept::Bool
    copy_X::Bool
    max_iter::Union{Int, Nothing}
    tol::Float64
    solver::Symbol
    positive::Bool
    random_state::Union{Int, Nothing}
    n_iter_::Union{Int, Nothing}
    fitted::Bool

    function Ridge(;
        α::Float64 = 1.0,
        fit_intercept::Bool = true,
        copy_X::Bool = true,
        max_iter::Union{Int, Nothing} = nothing,
        tol::Float64 = 1e-4,
        solver::Symbol = :auto,
        positive::Bool = false,
        random_state::Union{Int, Nothing} = nothing
    )
        new(
            Vector{Float64}(),
            0.0,
            α,
            fit_intercept,
            copy_X,
            max_iter,
            tol,
            solver,
            positive,
            random_state,
            nothing,
            false
        )
    end
end

function (model::Ridge)(X::AbstractMatrix{T}, y::AbstractVector{T}) where T <: Real
    n_samples, n_features = size(X)
    
    if model.copy_X
        X = copy(X)
    end
    
    if model.fit_intercept
        X_mean = mean(X, dims=1)
        y_mean = mean(y)
        X .-= X_mean
        y .-= y_mean
    end
    
    if model.solver == :auto
        model.solver = n_features > n_samples ? :lsqr : :cholesky
    end
    
    if model.solver == :cholesky
        A = X' * X + model.α * I
        b = X' * y
        model.w = A \ b
    elseif model.solver == :lsqr
        model.w, _ = lsqr(X, y, model.α, maxiter=model.max_iter, atol=model.tol, btol=model.tol)
    elseif model.solver == :sag || model.solver == :saga
        model.w = sag_solver(X, y, model.α, model.max_iter, model.tol, model.random_state, model.solver == :saga)
    elseif model.solver == :lbfgs
        if !model.positive
            error("LBFGS solver only supports positive=true")
        end
        opt = optimize(
            w -> ridge_loss(X, y, w, model.α),
            w -> ridge_gradient!(X, y, w, model.α),
            zeros(n_features),
            LBFGS(),
            Optim.Options(iterations=model.max_iter, g_tol=model.tol)
        )
        model.w = Optim.minimizer(opt)
        model.n_iter_ = Optim.iterations(opt)
    else
        error("Unknown solver: $(model.solver)")
    end
    
    if model.fit_intercept
        model.b = y_mean - dot(X_mean, model.w)
    else
        model.b = 0.0
    end
    
    model.fitted = true
    return model
end

function (model::Ridge)(X::AbstractMatrix{T}; type=nothing) where T <: Real
    if !model.fitted
        throw(ErrorException("This Ridge instance is not fitted yet. Call the model with training data before using it for predictions."))
    end
    
    if model.fit_intercept
        return X * model.w .+ model.b
    else
        return X * model.w
    end
end

function ridge_loss(X::AbstractMatrix, y::AbstractVector, w::AbstractVector, α::Float64)
    return sum(abs2, X * w - y) / 2 + α * sum(abs2, w) / 2
end

function ridge_gradient!(X::AbstractMatrix, y::AbstractVector, w::AbstractVector, α::Float64)
    return X' * (X * w - y) + α * w
end

function lsqr(A::AbstractMatrix, b::AbstractVector, α::Float64; maxiter::Int=nothing, atol::Float64=1e-6, btol::Float64=1e-6)
    m, n = size(A)
    x = zeros(n)
    v = zeros(n)
    u = copy(b)
    β = norm(u)
    u ./= β
    r = A' * u
    w = copy(r)
    
    ϕ_bar = β
    ρ_bar = norm(r)
    
    for iter in 1:maxiter
        q = A * w
        α_lsqr = ρ_bar / norm(q)
        u .-= α_lsqr .* q
        β = norm(u)
        u ./= β
        r = A' * u .- β .* v
        ρ = norm(r)
        
        c = ρ_bar / sqrt(ρ_bar^2 + β^2)
        s = β / sqrt(ρ_bar^2 + β^2)
        θ = s * α_lsqr
        ρ_bar = -c * α_lsqr
        ϕ = c * ϕ_bar
        ϕ_bar = s * ϕ_bar
        
        x .+= (ϕ / ρ) .* w
        w = r .- (θ / ρ) .* w
        
        if norm(A' * (b - A * x) - α * x) < atol * norm(A' * b) && norm(b - A * x) < btol * norm(b)
            break
        end
    end
    
    return x, iter
end

function sag_solver(X::AbstractMatrix, y::AbstractVector, α::Float64, max_iter::Int, tol::Float64, random_state::Union{Int, Nothing}, saga::Bool)
    n_samples, n_features = size(X)
    if random_state !== nothing
        Random.seed!(random_state)
    end
    
    w = zeros(n_features)
    grad_sum = zeros(n_features)
    memory = zeros(n_samples, n_features)
    
    for iter in 1:max_iter
        for i in randperm(n_samples)
            old_grad = memory[i, :]
            new_grad = X[i, :] * (dot(X[i, :], w) - y[i])
            if saga
                grad_sum .-= old_grad
                grad_sum .+= new_grad
                w .-= (new_grad .- old_grad .+ grad_sum ./ n_samples .+ α .* w) ./ (n_samples * α)
            else
                grad_sum .-= old_grad
                grad_sum .+= new_grad
                w .-= (grad_sum ./ n_samples .+ α .* w) ./ (n_samples * α)
            end
            memory[i, :] = new_grad
        end
        
        if norm(grad_sum ./ n_samples .+ α .* w) < tol
            break
        end
    end
    
    return w
end

function Base.show(io::IO, model::Ridge)
    println(io, "Ridge(")
    println(io, "  α=$(model.α),")
    println(io, "  fit_intercept=$(model.fit_intercept),")
    println(io, "  copy_X=$(model.copy_X),")
    println(io, "  max_iter=$(model.max_iter),")
    println(io, "  tol=$(model.tol),")
    println(io, "  solver=$(model.solver),")
    println(io, "  positive=$(model.positive),")
    println(io, "  random_state=$(model.random_state),")
    println(io, "  fitted=$(model.fitted)")
    print(io, ")")
end