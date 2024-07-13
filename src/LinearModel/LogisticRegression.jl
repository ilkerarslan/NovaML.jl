using Random, ProgressBars, Statistics, LinearAlgebra, Optim
import ...NovaML: AbstractModel, net_input, sigmoid

mutable struct LogisticRegression <: AbstractModel
    # Parameters
    w::Vector{Float64}
    b::Float64 
    losses::Vector{Float64}
    fitted::Bool 

    # Hyperparameters
    η::Float64
    num_iter::Int
    random_state::Union{Int, Nothing}
    solver::Symbol 
    batch_size::Int
    λ::Float64  # Parameter for regularization
    tol::Float64  # Tolerance for stopping criterion
    max_iter::Int  # Maximum number of iterations for L-BFGS
end

function LogisticRegression(; η=0.01, num_iter=100, random_state=nothing, solver=:lbfgs, batch_size=32, λ=1e-4, tol=1e-4, max_iter=100)
    if !(solver ∈ [:sgd, :batch, :minibatch, :lbfgs])
        throw(ArgumentError("`solver` should be in [:sgd, :batch, :minibatch, :lbfgs]"))
    else
        return LogisticRegression(Float64[], 0.0, Float64[], false, η, num_iter, random_state, solver, batch_size, λ, tol, max_iter)
    end
end

(m::LogisticRegression)(x::AbstractVector) = sigmoid(net_input(m, x)) ≥ 0.5 ? 1 : 0

(m::LogisticRegression)(X::AbstractMatrix) = [m(x) for x in eachrow(X)]

function (m::LogisticRegression)(X::AbstractMatrix, y::AbstractVector)
    if m.random_state !== nothing
        Random.seed!(m.random_state)
    end
    empty!(m.losses)    
    n_samples, n_features = size(X)

    if m.solver == :lbfgs
        # L-BFGS solver
        θ = vcat(zeros(n_features), 0.0)  # Initialize with zeros
        
        function f(θ)
            w, b = θ[1:end-1], θ[end]
            z = X * w .+ b
            ŷ = sigmoid.(z)
            loss = -mean(y .* log.(ŷ .+ eps()) .+ (1 .- y) .* log.(1 .- ŷ .+ eps())) + 0.5 * m.λ * sum(w.^2)
            return loss
        end
        
        function g!(G, θ)
            w, b = θ[1:end-1], θ[end]
            z = X * w .+ b
            ŷ = sigmoid.(z)
            G[1:end-1] = (X' * (ŷ .- y) ./ n_samples) .+ m.λ .* w
            G[end] = sum(ŷ .- y) / n_samples
        end
        
        res = optimize(f, g!, θ, LBFGS(), Optim.Options(iterations=m.max_iter, g_tol=m.tol))
        θ_opt = Optim.minimizer(res)
        m.w, m.b = θ_opt[1:end-1], θ_opt[end]
        m.losses = [Optim.minimum(res)]
    else
        # Initialize weights
        m.w = zeros(n_features)
        m.b = 0.0

        if m.solver == :batch
            # Batch Gradient Descent
            for _ in ProgressBar(1:m.num_iter)
                ŷ = sigmoid.(net_input(m, X))
                errors = y .- ŷ
                # Update weights and bias
                m.w .+= m.η .* (X' * errors .- m.λ .* m.w) ./ n_samples
                m.b += m.η .* sum(errors) / n_samples
                # Calculate and store the loss
                loss = -mean(y .* log.(ŷ .+ eps()) .+ (1 .- y) .* log.(1 .- ŷ .+ eps())) + 0.5 * m.λ * sum(m.w .^ 2)
                push!(m.losses, loss)
            end
        elseif m.solver == :sgd
            # Stochastic Gradient Descent
            for _ in ProgressBar(1:m.num_iter)
                for i in 1:n_samples
                    xi, yi = X[i, :], y[i]
                    ŷi = sigmoid(net_input(m, xi))
                    error = yi - ŷi
                    # Update weights and bias
                    m.w .+= m.η .* (error .* xi .- m.λ .* m.w)
                    m.b += m.η * error
                end
                # Calculate and store the loss
                ŷ = sigmoid.(net_input(m, X))
                loss = -mean(y .* log.(ŷ .+ eps()) .+ (1 .- y) .* log.(1 .- ŷ .+ eps())) + 0.5 * m.λ * sum(m.w .^ 2)
                push!(m.losses, loss)
            end
        elseif m.solver == :minibatch
            # Mini-Batch Gradient Descent
            for _ in ProgressBar(1:m.num_iter)
                shuffle_indices = Random.shuffle(1:n_samples)
                for batch_start in 1:m.batch_size:n_samples
                    batch_end = min(batch_start + m.batch_size - 1, n_samples)
                    batch_indices = shuffle_indices[batch_start:batch_end]
                    X_batch = X[batch_indices, :]
                    y_batch = y[batch_indices]
                    
                    ŷ_batch = sigmoid.(net_input(m, X_batch))
                    errors = y_batch .- ŷ_batch
                    # Update weights and bias
                    m.w .+= m.η .* (X_batch' * errors .- m.λ .* m.w) ./ length(batch_indices)
                    m.b += m.η .* sum(errors) / length(batch_indices)
                end
                # Calculate and store the loss
                ŷ = sigmoid.(net_input(m, X))
                loss = -mean(y .* log.(ŷ .+ eps()) .+ (1 .- y) .* log.(1 .- ŷ .+ eps())) + 0.5 * m.λ * sum(m.w .^ 2)
                push!(m.losses, loss)
            end
        end
    end
    m.fitted = true
end