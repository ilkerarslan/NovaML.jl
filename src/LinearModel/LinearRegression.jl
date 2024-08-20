using LinearAlgebra
using Statistics

import ...NovaML: AbstractModel

mutable struct LinearRegression <: AbstractModel
    w::Vector{Float64}
    b::Float64
    fit_intercept::Bool
    normalize::Bool
    copy_X::Bool    
    positive::Bool
    solver::Symbol
    η::Float64  
    num_iter::Int  
    tol::Float64  
    fitted::Bool
    losses::Vector{Float64}  

    function LinearRegression(;
        fit_intercept::Bool=true,
        normalize::Bool=false,
        copy_X::Bool=true,
        positive::Bool=false,
        solver::Symbol=:normal,
        η::Float64=0.01,
        num_iter::Int=1000,
        tol::Float64=1e-4
    )
        @assert solver in [:normal, :batch] "solver must be either :normal or :batch"
        new(
            Vector{Float64}(),
            0.0,
            fit_intercept,
            normalize,
            copy_X,
            positive,
            solver,
            η,
            num_iter,
            tol,
            false,
            Float64[]
        )
    end
end

function (model::LinearRegression)(X::AbstractVecOrMat{T}, y::AbstractVector{T}) where T <: Real
    X_matrix = X isa AbstractVector ? reshape(X, :, 1) : X
    n_samples, n_features = size(X_matrix)
    
    if model.copy_X
        X_matrix = copy(X_matrix)
    end
    
    if model.normalize
        X_mean = mean(X_matrix, dims=1)
        X_std = std(X_matrix, dims=1)
        X_matrix = (X_matrix .- X_mean) ./ X_std
    end
    
    if model.fit_intercept
        X_matrix = hcat(ones(n_samples), X_matrix)
        n_features += 1
    end
    
    empty!(model.losses)  
    
    if model.solver == :normal
        if model.positive        
            model.w = nnls(X_matrix, y)
        else            
            model.w = X_matrix \ y
        end
        
        y_pred = X_matrix * model.w
        mse = mean((y_pred - y).^2)
        push!(model.losses, mse)
    elseif model.solver == :batch
        model.w = zeros(n_features)        
        for _ in 1:model.num_iter
            y_pred = X_matrix * model.w
            gradient = (2/n_samples) * X_matrix' * (y_pred - y)
            model.w -= model.η * gradient
            
            if model.positive
                model.w = max.(model.w, 0)
            end            
            mse = mean((y_pred - y).^2)
            push!(model.losses, mse)
            
            if length(model.losses) > 1 && abs(model.losses[end-1] - mse) < model.tol
                break
            end
        end
    end
    
    if model.fit_intercept
        model.b = model.w[1]
        model.w = model.w[2:end]
    else
        model.b = 0.0
    end
    
    model.fitted = true
    return model
end

function (model::LinearRegression)(X::AbstractVecOrMat{T}) where T <: Real
    if !model.fitted
        throw(ErrorException("This LinearRegression instance is not fitted yet. Call the model with training data before using it for predictions."))
    end
    
    X_matrix = X isa AbstractVector ? reshape(X, :, 1) : X
    
    if model.normalize
        X_mean = mean(X_matrix, dims=1)
        X_std = std(X_matrix, dims=1)
        X_matrix = (X_matrix .- X_mean) ./ X_std
    end
    
    return X_matrix * model.w .+ model.b
end

function nnls(X::AbstractMatrix{T}, y::AbstractVector{T}) where T <: Real
    n_features = size(X, 2)
    w = zeros(n_features)
    residual = y - X * w
    
    for _ in 1:1000  # Max iterations
        gradient = X' * residual
        if all(w .> 0) && all(gradient .<= 0)
            break
        end
        
        index = argmax(gradient)
        w[index] += gradient[index]
        w = max.(w, 0)
        residual = y - X * w
    end
    
    return w
end

function Base.show(io::IO, model::LinearRegression)
    println(io, "LinearRegression(")
    println(io, "  fit_intercept=$(model.fit_intercept),")
    println(io, "  normalize=$(model.normalize),")
    println(io, "  copy_X=$(model.copy_X),")
    println(io, "  positive=$(model.positive),")
    println(io, "  solver=$(model.solver),")
    println(io, "  η=$(model.η),")
    println(io, "  num_iter=$(model.num_iter),")
    println(io, "  tol=$(model.tol),")
    println(io, "  fitted=$(model.fitted),")
    println(io, "  n_losses=$(length(model.losses))")
    print(io, ")")
end