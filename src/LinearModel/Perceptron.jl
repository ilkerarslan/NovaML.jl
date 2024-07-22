using Random, ProgressBars
import ...NovaML: AbstractModel, net_input

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
    solver::Symbol
    batch_size::Int 
end

function Perceptron(; η=0.01, num_iter=100, random_state=nothing,
            solver=:sgd, batch_size=32)
    if !(solver ∈ [:sgd, :batch, :minibatch])
        throw(ArgumentError("`solver` should be in [:sgd, :batch, :minibatch]"))
    else        
        return Perceptron(Float64[], 0.0, Float64[], false, η, num_iter, random_state, solver, batch_size)
    end
end

function (m::Perceptron)(X::Matrix, y::Vector)
    if m.random_state !== nothing
        Random.seed!(m.random_state)
    end
    empty!(m.losses)


    # Initialize weights
    m.w = randn(size(X, 2)) ./ 100
    n = length(y)
    if m.solver == :sgd        
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
    elseif m.solver == :batch
        for _ ∈ ProgressBar(1:m.num_iter)
            ŷ = [m(X[i, :]) for i in 1:n]
            ∇ = m.η * (y - ŷ)
            m.w .+= X' * ∇
            m.b += sum(∇)
            error = sum(abs.(∇))
            push!(m.losses, error)
        end
    elseif m.solver == :minibatch
        num_batches = ceil(Int, n / m.batch_size)
        for _ ∈ ProgressBar(1:m.num_iter)
            error = 0
            shuffle_indices = Random.shuffle(1:n)
            for batch in Iterators.partition(shuffle_indices, m.batch_size)
                X_batch = X[batch, :]
                y_batch = y[batch]
                ŷ_batch = [m(X_batch[i, :]) for i in 1:length(batch)]
                ∇ = m.η * (y_batch - ŷ_batch)
                m.w .+= X_batch' * ∇
                m.b += sum(∇)
                error += sum(abs.(∇))
            end
            push!(m.losses, error)
        end
    end
    m.fitted = true
end

(m::Perceptron)(x::AbstractVector) = net_input(m, x) ≥ 0.0 ? 1 : 0