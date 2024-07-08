module PerceptronModel

using Random, ProgressBars
import ...Nova: AbstractModel, net_input

export Perceptron 

"""
    Perceptron <: AbstractModel

Perceptron model for binary classification.

# Fields
- `w::Vector{Float64}`: Weight vector
- `b::Float64`: Bias term
- `losses::Vector{Float64}`: Training losses
- `fitted::Bool`: Indicates if the model has been fitted
- `η::Float64`: Learning rate
- `num_iter::Int`: Number of iterations for training
- `random_state::Union{Nothing, Int64}`: Seed for random number generator
- `optim_alg::Symbol`: Optimization algorithm (:SGD, :Batch, or :MiniBatch)
- `batch_size::Int`: Batch size for mini-batch optimization
"""
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

"""
    Perceptron(; η=0.01, num_iter=100, random_state=nothing, optim_alg=:SGD, batch_size=32)

Constructor for Perceptron model.

# Keywords
- `η::Float64=0.01`: Learning rate
- `num_iter::Int=100`: Number of iterations for training
- `random_state::Union{Nothing, Int64}=nothing`: Seed for random number generator
- `optim_alg::Symbol=:SGD`: Optimization algorithm (:SGD, :Batch, or :MiniBatch)
- `batch_size::Int=32`: Batch size for mini-batch optimization

# Returns
- `Perceptron`: Initialized Perceptron model
"""
function Perceptron(; η=0.01, num_iter=100, random_state=nothing,
            optim_alg=:SGD, batch_size=32)
    if !(optim_alg ∈ [:SGD, :Batch, :MiniBatch])
        throw(ArgumentError("`optim_alg` should be in [:SGD, :Batch, :MiniBatch]"))
    else        
        return Perceptron(Float64[], 0.0, Float64[], false, η, num_iter, random_state, optim_alg, batch_size)
    end
end

"""
    (m::Perceptron)(X::Matrix, y::Vector)

Train the Perceptron model.

# Arguments
- `X::Matrix`: Input feature matrix, where each row is a sample
- `y::Vector`: Target labels

# Effects
- Updates the model parameters (`w` and `b`)
- Stores the training losses in `m.losses`
- Sets `m.fitted` to `true`
"""
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
        # TODO: Implement batch gradient descent
    elseif m.optim_alg == :MiniBatch
        # TODO: Implement mini-batch gradient descent
    end
end

"""
    (m::Perceptron)(x::AbstractVector)

Predict class label for a single sample.

# Arguments
- `x::AbstractVector`: Input feature vector

# Returns
- `Int`: Predicted class label (0 or 1)
"""
(m::Perceptron)(x::AbstractVector) = net_input(m, x) ≥ 0.0 ? 1 : 0

end