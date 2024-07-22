using Random, Statistics, ProgressBars
import ...NovaML: AbstractModel, linearactivation, net_input 

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
    solver::Symbol
    batch_size::Int
    w_init::Bool
    shuffle::Bool
end


function Adaline(; η=0.01, num_iter=100, random_state=nothing,
                   solver=:sgd, batch_size=32, shuffle=true)
    if !(solver ∈ [:sgd, :batch, :minibatch])
        throw(ArgumentError("`solver` should be in [:sgd, :batch, :minibatch]"))
    else        
        return Adaline(Float64[], 0.0, Float64[], false, η, num_iter, random_state, solver, batch_size, false, shuffle)
    end
end

(m::Adaline)(x::AbstractVector) = linearactivation(net_input(m, x)) ≥ 0.5 ? 1 : 0

function (m::Adaline)(X::Matrix, y::Vector; partial=false)
    if m.random_state !== nothing
        Random.seed!(m.random_state) 
    end
    
    if !m.w_init
        m.w = randn(size(X,2)) ./ 100
        m.w_init = true
    end
    
    empty!(m.losses)
    n, p = size(X)

    if m.solver == :sgd
        for _ ∈ 1:m.num_iter
            if m.shuffle
                idx = randperm(n)
                X, y = X[idx, :], y[idx]
            end
            losses = Float64[]
            for i ∈ 1:n
                ŷ = linearactivation(net_input(m, X[i, :]))
                error = (y[i] - ŷ)
                m.w .+= m.η * 2.0 .* X[i, :] .* error
                m.b += m.η * 2.0 * error
                loss = error^2
                push!(losses, loss)
            end
            avg_loss = mean(losses)
            push!(m.losses, avg_loss)
        end
    elseif m.solver == :batch
        for i ∈ ProgressBar(1:m.num_iter)
            ŷ = linearactivation(net_input(m, X))
            errors = (y .- ŷ)
            m.w .+= m.η .* 2.0 .* X'*errors ./ n
            m.b += m.η * 2.0 * mean(errors)
            loss = mean(errors.^2)
            push!(m.losses, loss)
        end
    elseif m.solver == :minibatch
        for _ ∈ ProgressBar(1:m.num_iter)
            if m.shuffle
                idx = randperm(n)
                X, y = X[idx, :], y[idx]
            end
            batch_losses = Float64[]
            for batch_start in 1:m.batch_size:n
                batch_end = min(batch_start + m.batch_size - 1, n)
                X_batch = X[batch_start:batch_end, :]
                y_batch = y[batch_start:batch_end]
                
                ŷ_batch = linearactivation(net_input(m, X_batch))
                errors = y_batch .- ŷ_batch
                
                m.w .+= m.η .* 2.0 .* X_batch'*errors ./ length(y_batch)
                m.b += m.η * 2.0 * mean(errors)
                
                batch_loss = mean(errors.^2)
                push!(batch_losses, batch_loss)
            end
            avg_loss = mean(batch_losses)
            push!(m.losses, avg_loss)
        end
    end
    
    m.fitted = true
end