module LinearModel

using Random, Statistics, ProgressBars

export Perceptron

abstract type AbstractModel end

# Common methods 
"""Calculate net input"""
net_input(m::AbstractModel, x::AbstractVector) = x'*m.w + m.b
net_input(m::AbstractModel, X::AbstractMatrix) = X * m.w .+ m.b

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
    optim_alg::String
    batch_size::Int 
end

function Perceptron(; η=0.01, num_iter=100, random_state=nothing,
            optim_alg="SGD", batch_size=32)
    if !(optim_alg ∈ ["SGD", "Batch", "MiniBatch"])
        throw("""`optim_alg` should be in [SGD", "Batch", "MiniBatch"]""")
    else        
        return Perceptron(Float64[], randn()/100, Float64[], false, η, num_iter, random_state, optim_alg, batch_size)
    end
end

# Prediction
"""Perceptron prediction with single observation"""
(m::Perceptron)(x::AbstractVector) = (x'*m.w + m.b) ≥ 0.0 ? 1 : 0
"""Perceptron batch prediction"""
(m::Perceptron)(X::AbstractMatrix) = [m(x) for x in eachrow(X)]

# Model training
"""Perceptron training model"""
function (m::Perceptron)(X::Matrix, y::Vector)
    if m.random_state !== nothing
        Random.seed!(m.random_state)
    end
    empty!(m.losses)

    # Initialize weights
    m.w = randn(size(X, 2)) ./ 100

    if m.optim_alg == "SGD"
        n = length(y)
        for _ in ProgressBar(1:m.num_iter)
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
    elseif m.optim_alg == "Batch"
        # Implement batch gradient descent
    elseif m.optim_alg == "MiniBatch"
        # Implement mini-batch gradient descent
    end
end


end # of module LinearModel