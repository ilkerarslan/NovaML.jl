module LinearModel

using Random, Statistics, ProgressBars

export Perceptron, Adaline, MulticlassPerceptron
export LogisticRegression, MultinomialLogisticRegression

abstract type AbstractModel end
abstract type AbstractMultiClass <: AbstractModel end

# Activation functions
"""Linear activation function"""
linearactivation(X) = X
"""Sigmoid activation function for scalar value"""
sigmoid(z::Real) = 1 / (1 + exp(-z))
"""Sigmoid activation function for an array"""
sigmoid(z::AbstractArray) = 1 ./ (1 .+ exp.(-z))

"""Compute logistic sigmoid activation"""
sigmoid(z) = 1. / (1. + exp(-1*clamp(z, -250, 250)))

# Common methods
"""Calculate net input for AbstractModel"""
net_input(m::AbstractModel, x::AbstractVector) = x'*m.w + m.b
net_input(m::AbstractModel, X::AbstractMatrix) = X * m.w .+ m.b
# net_input for MulticlassPerceptron
"""Calculate net input for AbstractMultiClass"""
net_input(m::AbstractMultiClass, x::AbstractVector) = m.W' * x .+ m.b
net_input(m::AbstractMultiClass, X::AbstractMatrix) = X * m.W .+ m.b'

"""AbstractModel batch prediction"""
(m::AbstractModel)(X::AbstractMatrix) = [m(x) for x in eachrow(X)]

#Softmax function for probability distribution
function softmax(X::AbstractMatrix)
    exp_X = exp.(X .- maximum(X, dims=2))
    return exp_X ./ sum(exp_X, dims=2)
end

########################################################
# Perceptron
mutable struct Perceptron <: AbstractModel
    # Parameters
    w::Vector{Float64}
    b::Float64
    losses::Vector{Float64}
    fitted::Bool

    # Hyper parameters
    η::Float64
    num_iter::Int
    random_state::Union{Nothing, Int64}
    optim_alg::Symbol
    batch_size::Int 
end

function Perceptron(; η=0.01, num_iter=100, random_state=nothing,
            optim_alg=:SGD, batch_size=32)
    if !(optim_alg ∈ [:SGD, :Batch, :MiniBatch])
        throw("""`optim_alg` should be in [:SGD, :Batch, :MiniBatch]""")
    else        
        return Perceptron(Float64[], 0.0, Float64[], false, η, num_iter, random_state, optim_alg, batch_size)
    end
end

# Model training
"""Perceptron training model"""
function (m::Perceptron)(X::Matrix, y::Vector)
    if m.random_state !== nothing
        Random.seed!(m.random_state)
    end
    empty!(m.losses)

    # Initialize weights
    m.w = randn(size(X, 2)) ./ 100

    if m.optim_alg == :SGD
        n = length(y)
        for _ ∈ ProgressBar(1:m.num_iter)
            error = 0
            for i in 1:n
                xi, yi = X[i, :], y[i]
                ŷ = m(xi)                
                ∇ = m.η * (yi - ŷ)
                m.w .+= ∇ * xi
                m.b += ∇
                error += Int(∇ != 0.0)
            end
            push!(m.losses, error)
        end
        m.fitted = true
    elseif m.optim_alg == :Batch
        # Implement batch gradient descent
    elseif m.optim_alg == :MiniBatch
        # Implement mini-batch gradient descent
    end
end

"""Perceptron prediction with single observation"""
(m::Perceptron)(x::AbstractVector) = net_input(m, x) ≥ 0.0 ? 1 : 0

########################################################
# Adaline
"""Adaline training model"""
mutable struct Adaline <: AbstractModel
    # Parameters
    w::Vector{Float64}
    b::Float64
    losses::Vector{Float64}
    fitted::Bool

    # Hyper parameters
    η::Float64
    num_iter::Int
    random_state::Union{Nothing, Int64}
    optim_alg::Symbol
    batch_size::Int
    w_init::Bool
    shuffle::Bool
end

function Adaline(; η=0.01, num_iter=100, random_state=nothing,
            optim_alg=:SGD, batch_size=32, shuffle=true)
    if !(optim_alg ∈ [:SGD, :Batch, :MiniBatch])
        throw("""`optim_alg` should be in [:SGD, :Batch, :MiniBatch]""")
    else        
        return Adaline(Float64[], 0.0, Float64[], false, η, num_iter, random_state, optim_alg, batch_size, false, shuffle)
    end
end

"""Adaline prediction with single observation"""
(m::Adaline)(x::AbstractVector) = linearactivation(net_input(m, x)) ≥ 0.5 ? 1 : 0

function (m::Adaline)(X::Matrix, y::Vector; partial=false)
    if m.optim_alg == :SGD
        if partial == true
            println("Partial fit for Adaline hasn't been implemented yet")
        else
            if m.random_state !== nothing
               Random.seed!(m.random_state) 
            end
            m.w = randn(size(X,2)) ./ 100
            m.w_init = true
            empty!(m.losses)
            for _ ∈ 1:m.num_iter
                if m.shuffle
                    idx = randperm(length(y))
                    X, y = X[idx, :], y[idx]
                end
                losses = []
                for i ∈ 1:size(X, 1)
                    ŷ = linearactivation(net_input(m, X[i, :]))
                    error = (y[i] - ŷ)
                    m.w .+= m.η * 2.0 .* X[i, :] .* error
                    m.b += m.η * 2.0 *error
                    loss = error^2
                    push!(losses, loss)
                end
                avg_loss = mean(losses)
                push!(m.losses, avg_loss)
            end
        end
    elseif m.optim_alg == :Batch
        if m.random_state !== nothing
            Random.seed!(m.random_state)
        end
        empty!(m.losses)
    
        # Initialize weights
        m.w = randn(size(X, 2)) ./ 100

        n = length(y)
        for i ∈ ProgressBar(1:m.num_iter)
            ŷ = linearactivation(net_input(m, X))
            errors = (y .- ŷ)
            m.w .+= m.η .* 2.0 .* X'*errors ./ n
            m.b += m.η * 2.0 * mean(errors)
            loss = mean(errors.^2)
            push!(m.losses, loss)
        end
    elseif m.optim_alg == :MiniBatch
        println("MiniBatch algorithm for Adaline hasn't been implemented yet.")
    end
end

########################################################################
# MulticlassPerceptron
mutable struct MulticlassPerceptron <: AbstractMultiClass
    # Parameters
    W::Matrix{Float64}
    b::Vector{Float64}
    losses::Vector{Float64}
    fitted::Bool
    classes::Vector{Any}

    # Hyper parameters
    η::Float64
    num_iter::Int
    random_state::Union{Nothing, Int64}
    optim_alg::Symbol
    batch_size::Int 
end

function MulticlassPerceptron(; η=0.01, num_iter=100, random_state=nothing,
            optim_alg=:SGD, batch_size=32)
    if !(optim_alg ∈ [:SGD, :Batch, :MiniBatch])
        throw("""`optim_alg` should be in [:SGD, :Batch, :MiniBatch]""")
    else        
        return MulticlassPerceptron(Matrix{Float64}(undef, 0, 0), Float64[], 
                                    Float64[], false, [], η, num_iter, 
                                    random_state, optim_alg, batch_size)
    end
end

# Model training
"""Multiclass Perceptron training model"""
function (m::MulticlassPerceptron)(X::Matrix, y::Vector)
    if m.optim_alg == :SGD
        if m.random_state !== nothing
            Random.seed!(m.random_state)
        end
        empty!(m.losses)
    
        # Get unique classes
        m.classes = sort(unique(y))
        n_classes = length(m.classes)
        n_features = size(X, 2)
    
        # Initialize weights and bias
        m.W = randn(n_features, n_classes) ./ 100
        m.b = zeros(n_classes)
    
        # Create a dictionary to map classes to indices
        class_to_index = Dict(class => i for (i, class) in enumerate(m.classes))
    
    
        n = length(y)
        for _ in 1:m.num_iter
            error = 0
            for i in 1:n
                xi, yi = X[i, :], y[i]
                ŷ_scores = net_input(m, xi)
                ŷ_index = argmax(ŷ_scores)
                yi_index = class_to_index[yi]
                if ŷ_index != yi_index
                    error += 1
                    # Update weights and bias for the correct class
                    m.W[:, yi_index] .+= m.η * xi
                    m.b[yi_index] += m.η
                    # Update weights and bias for the predicted class
                    m.W[:, ŷ_index] .-= m.η * xi
                    m.b[ŷ_index] -= m.η
                end
            end
            push!(m.losses, error)
        end
        m.fitted = true
    elseif m.optim_alg == :Batch
        # Implement batch gradient descent
    elseif m.optim_alg == :MiniBatch
        # Implement mini-batch gradient descent
    end
end

"""Multiclass Perceptron prediction with single observation"""
function (m::MulticlassPerceptron)(x::AbstractVector)
    if !m.fitted
        throw(ErrorException("Model is not fitted yet."))
    end
    scores = net_input(m, x)
    class_index = argmax(scores)
    return m.classes[class_index]
end

"""Multiclass Perceptron prediction with multiple observations"""
function (m::MulticlassPerceptron)(X::AbstractMatrix)
    if !m.fitted
        throw(ErrorException("Model is not fitted yet."))
    end
    scores = net_input(m, X)
    class_indices = [argmax(score) for score in eachrow(scores)]
    return [m.classes[i] for i in class_indices]
end

#######################################################
# Logistic Regression

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
    λ::Float64  # PArameter for regularization
end

function LogisticRegression(; η=0.01, num_iter=50, random_state=nothing, optim_alg=:SGD, batch_size=32, λ=0.01)
    if !(optim_alg ∈ [:SGD, :Batch, :MiniBatch])
        throw(ArgumentError("`optim_alg` should be in [:SGD, :Batch, :MiniBatch]"))
    else
        return LogisticRegression(Float64[], 0.0, Float64[], false, η, num_iter, random_state, optim_alg, batch_size, λ)
    end
end

"""Predict class label for a single sample"""
(m::LogisticRegression)(x::AbstractVector) = sigmoid(net_input(m, x)) ≥ 0.5 ? 1 : 0

"""Predict class labels for multiple samples"""
(m::LogisticRegression)(X::AbstractMatrix) = [m(x) for x in eachrow(X)]

"""Logistic Regression training"""
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
        for _ in ProgressBar(1:m.num_iter)
            ŷ = sigmoid.(net_input(m, X))
            errors = y .- ŷ
            m.w .+= m.η .* (X' * errors .- m.λ .* m.w) ./ n_samples
            m.b += m.η .* sum(errors) / n_samples
            loss = -mean(y .* log.(ŷ .+ eps()) .+ (1 .- y) .* log.(1 .- ŷ .+ eps())) + 0.5 * m.λ * sum(m.w .^ 2)
            push!(m.losses, loss)
        end
    elseif m.optim_alg == :SGD
        for _ in ProgressBar(1:m.num_iter)
            for i in 1:n_samples
                xi, yi = X[i, :], y[i]
                ŷi = sigmoid(net_input(m, xi))
                error = yi - ŷi
                m.w .+= m.η .* (error .* xi .- m.λ .* m.w)
                m.b += m.η * error
            end
            ŷ = sigmoid.(net_input(m, X))
            loss = -mean(y .* log.(ŷ .+ eps()) .+ (1 .- y) .* log.(1 .- ŷ .+ eps())) + 0.5 * m.λ * sum(m.w .^ 2)
            push!(m.losses, loss)
        end
    elseif m.optim_alg == :MiniBatch
        for _ in ProgressBar(1:m.num_iter)
            shuffle_indices = Random.shuffle(1:n_samples)
            for batch_start in 1:m.batch_size:n_samples
                batch_end = min(batch_start + m.batch_size - 1, n_samples)
                batch_indices = shuffle_indices[batch_start:batch_end]
                X_batch = X[batch_indices, :]
                y_batch = y[batch_indices]
                
                ŷ_batch = sigmoid.(net_input(m, X_batch))
                errors = y_batch .- ŷ_batch
                m.w .+= m.η .* (X_batch' * errors .- m.λ .* m.w) ./ length(batch_indices)
                m.b += m.η .* sum(errors) / length(batch_indices)
            end
            ŷ = sigmoid.(net_input(m, X))
            loss = -mean(y .* log.(ŷ .+ eps()) .+ (1 .- y) .* log.(1 .- ŷ .+ eps())) + 0.5 * m.λ * sum(m.w .^ 2)
            push!(m.losses, loss)
        end
    end
    m.fitted = true
end


end # of module LinearModel