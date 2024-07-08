module AdalineModel

using Random, Statistics, ProgressBars
import ...Nova: AbstractModel, linearactivation, net_input 

export Adaline 


"""
    Adaline <: AbstractModel

Adaptive Linear Neuron (ADALINE) model for binary classification.

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
- `w_init::Bool`: Indicates if weights have been initialized
- `shuffle::Bool`: Whether to shuffle data before each epoch
"""
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

"""
    Adaline(; η=0.01, num_iter=100, random_state=nothing, optim_alg=:SGD, batch_size=32, shuffle=true)

Constructor for Adaline model.

# Keywords
- `η::Float64=0.01`: Learning rate
- `num_iter::Int=100`: Number of iterations for training
- `random_state::Union{Nothing, Int64}=nothing`: Seed for random number generator
- `optim_alg::Symbol=:SGD`: Optimization algorithm (:SGD, :Batch, or :MiniBatch)
- `batch_size::Int=32`: Batch size for mini-batch optimization
- `shuffle::Bool=true`: Whether to shuffle data before each epoch

# Returns
- `Adaline`: Initialized Adaline model
"""
function Adaline(; η=0.01, num_iter=100, random_state=nothing,
            optim_alg=:SGD, batch_size=32, shuffle=true)
    if !(optim_alg ∈ [:SGD, :Batch, :MiniBatch])
        throw(ArgumentError("`optim_alg` should be in [:SGD, :Batch, :MiniBatch]"))
    else        
        return Adaline(Float64[], 0.0, Float64[], false, η, num_iter, random_state, optim_alg, batch_size, false, shuffle)
    end
end

"""
    (m::Adaline)(x::AbstractVector)

Predict class label for a single sample.

# Arguments
- `x::AbstractVector`: Input feature vector

# Returns
- `Int`: Predicted class label (0 or 1)
"""
(m::Adaline)(x::AbstractVector) = linearactivation(net_input(m, x)) ≥ 0.5 ? 1 : 0

"""
    (m::Adaline)(X::Matrix, y::Vector; partial=false)

Train the Adaline model.

# Arguments
- `X::Matrix`: Input feature matrix, where each row is a sample
- `y::Vector`: Target labels
- `partial::Bool=false`: If true, perform partial fit (not implemented yet)

# Effects
- Updates the model parameters (`w` and `b`)
- Stores the training losses in `m.losses`
- Sets `m.fitted` to `true`
"""
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
                    ŷ = linearactivation(net_input(m, X[i, :]))
                    error = (y[i] - ŷ)
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
            ŷ = linearactivation(net_input(m, X))
            errors = (y .- ŷ)
            m.w .+= m.η .* 2.0 .* X'*errors ./ n
            m.b += m.η * 2.0 * mean(errors)
            loss = mean(errors.^2)
            push!(m.losses, loss)
        end
    elseif m.optim_alg == :MiniBatch
        println("MiniBatch algorithm for Adaline hasn't been implemented yet.")
    end
end

end