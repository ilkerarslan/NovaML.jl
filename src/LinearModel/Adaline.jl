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
    if m.solver == :sgd
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
    elseif m.solver == :batch
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
    elseif m.solver == :minibatch
        println("MiniBatch algorithm for Adaline hasn't been implemented yet.")
    end
end