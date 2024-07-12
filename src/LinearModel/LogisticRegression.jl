module LogisticRegressionModel 

using Random, ProgressBars, Statistics
import ...Nova: AbstractModel, net_input, sigmoid

export LogisticRegression

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
    optim_alg::Symbol 
    batch_size::Int
    λ::Float64  # Parameter for regularization
end

function LogisticRegression(; η=0.01, num_iter=50, random_state=nothing, optim_alg=:SGD, batch_size=32, λ=0.01)
    if !(optim_alg ∈ [:SGD, :Batch, :MiniBatch])
        throw(ArgumentError("`optim_alg` should be in [:SGD, :Batch, :MiniBatch]"))
    else
        return LogisticRegression(Float64[], 0.0, Float64[], false, η, num_iter, random_state, optim_alg, batch_size, λ)
    end
end

(m::LogisticRegression)(x::AbstractVector) = sigmoid(net_input(m, x)) ≥ 0.5 ? 1 : 0

(m::LogisticRegression)(X::AbstractMatrix) = [m(x) for x in eachrow(X)]

function (m::LogisticRegression)(X::AbstractMatrix, y::AbstractVector)
    if m.random_state !== nothing
        Random.seed!(m.random_state)
    end
    empty!(m.losses)    
    # Initialize weights
    m.w = randn(size(X, 2)) ./ 100
    m.b = 0.0
    n_samples, n_features = size(X)

    if m.optim_alg == :Batch
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
    elseif m.optim_alg == :SGD
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
    elseif m.optim_alg == :MiniBatch
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
    m.fitted = true
end


end # of module LogisticRegressionModel