using Random, ProgressBars
import ...NovaML: AbstractModel, AbstractMultiClass, net_input 


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
        throw(ArgumentError("`optim_alg` should be in [:SGD, :Batch, :MiniBatch]"))
    else        
        return MulticlassPerceptron(Matrix{Float64}(undef, 0, 0), Float64[], 
                                    Float64[], false, [], η, num_iter, 
                                    random_state, optim_alg, batch_size)
    end
end

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

function (m::MulticlassPerceptron)(x::AbstractVector)
    if !m.fitted
        throw(ErrorException("Model is not fitted yet."))
    end
    scores = net_input(m, x)
    class_index = argmax(scores)
    return m.classes[class_index]
end

function (m::MulticlassPerceptron)(X::AbstractMatrix)
    if !m.fitted
        throw(ErrorException("Model is not fitted yet."))
    end
    scores = net_input(m, X)
    class_indices = [argmax(score) for score in eachrow(scores)]
    return [m.classes[i] for i in class_indices]
end