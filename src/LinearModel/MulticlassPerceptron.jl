module MulciclassPerceptronModel

using Random, ProgressBars
import ...Nova: AbstractModel, AbstractMultiClass, net_input 


export MulticlassPerceptron

"""
    MulticlassPerceptron <: AbstractMultiClass

Multiclass Perceptron model for classification tasks with more than two classes.

# Fields
- `W::Matrix{Float64}`: Weight matrix
- `b::Vector{Float64}`: Bias vector
- `losses::Vector{Float64}`: Training losses
- `fitted::Bool`: Indicates if the model has been fitted
- `classes::Vector{Any}`: Unique class labels
- `η::Float64`: Learning rate
- `num_iter::Int`: Number of iterations for training
- `random_state::Union{Nothing, Int64}`: Seed for random number generator
- `optim_alg::Symbol`: Optimization algorithm (:SGD, :Batch, or :MiniBatch)
- `batch_size::Int`: Batch size for mini-batch optimization
"""
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

"""
    MulticlassPerceptron(; η=0.01, num_iter=100, random_state=nothing, optim_alg=:SGD, batch_size=32)

Constructor for MulticlassPerceptron model.

# Keywords
- `η::Float64=0.01`: Learning rate
- `num_iter::Int=100`: Number of iterations for training
- `random_state::Union{Nothing, Int64}=nothing`: Seed for random number generator
- `optim_alg::Symbol=:SGD`: Optimization algorithm (:SGD, :Batch, or :MiniBatch)
- `batch_size::Int=32`: Batch size for mini-batch optimization

# Returns
- `MulticlassPerceptron`: Initialized MulticlassPerceptron model
"""
function MulticlassPerceptron(; η=0.01, num_iter=100, random_state=nothing,
            optim_alg=:SGD, batch_size=32)
    if !(optim_alg ∈ [:SGD, :Batch, :MiniBatch])
        throw(ArgumentError("`optim_alg` should be in [:SGD, :Batch, :MiniBatch]"))
    else        
        return MulticlassPerceptron(Matrix{Float64}(undef, 0, 0), Float64[], 
                                    Float64[], false, [], η, num_iter, 
                                    random_state, optim_alg, batch_size)
    end
end

"""
    (m::MulticlassPerceptron)(X::Matrix, y::Vector)

Train the MulticlassPerceptron model.

# Arguments
- `X::Matrix`: Input feature matrix, where each row is a sample
- `y::Vector`: Target labels

# Effects
- Updates the model parameters (`W` and `b`)
- Stores the training losses in `m.losses`
- Sets `m.fitted` to `true`
"""
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
        # TODO: Implement batch gradient descent
    elseif m.optim_alg == :MiniBatch
        # TODO: Implement mini-batch gradient descent
    end
end

"""
    (m::MulticlassPerceptron)(x::AbstractVector)

Predict class label for a single sample.

# Arguments
- `x::AbstractVector`: Input feature vector

# Returns
- `Any`: Predicted class label

# Throws
- `ErrorException`: If the model is not fitted yet
"""
function (m::MulticlassPerceptron)(x::AbstractVector)
    if !m.fitted
        throw(ErrorException("Model is not fitted yet."))
    end
    scores = net_input(m, x)
    class_index = argmax(scores)
    return m.classes[class_index]
end

"""
    (m::MulticlassPerceptron)(X::AbstractMatrix)

Predict class labels for multiple samples.

# Arguments
- `X::AbstractMatrix`: Input feature matrix, where each row is a sample

# Returns
- `Vector`: Predicted class labels

# Throws
- `ErrorException`: If the model is not fitted yet
"""
function (m::MulticlassPerceptron)(X::AbstractMatrix)
    if !m.fitted
        throw(ErrorException("Model is not fitted yet."))
    end
    scores = net_input(m, X)
    class_indices = [argmax(score) for score in eachrow(scores)]
    return [m.classes[i] for i in class_indices]
end

end # of module MulciclassPerceptronModel