using Optim
using LinearAlgebra
using Distances
using Statistics

mutable struct SVC
    kernel::Symbol
    C::Float64
    gamma::Union{Float64,Symbol}
    class_weight::Union{Dict{Int,Float64},Symbol}
    
    # Model parameters
    support_vectors::Matrix{Float64}
    dual_coef::Vector{Float64}
    intercept::Float64
    classes::Vector{Int}
    
    function SVC(;
        kernel::Symbol=:rbf,
        C::Float64=1.0,
        gamma::Union{Float64,Symbol}=:scale,
        class_weight::Union{Dict{Int,Float64},Symbol}=:balanced
    )
        new(kernel, C, gamma, class_weight)
    end
end

function kernel_function(X::Matrix{Float64}, Y::Matrix{Float64}, kernel::Symbol, gamma::Union{Float64,Symbol})
    if kernel == :linear
        return X * Y'
    elseif kernel == :rbf
        gamma_value = (gamma == :scale) ? 1.0 / size(X, 2) : gamma
        return exp.(-gamma_value .* max.(pairwise(SqEuclidean(), X', Y', dims=2), 0.0))
    else
        error("Unsupported kernel: $kernel")
    end
end

function (svc::SVC)(X::Matrix{Float64}, y::Vector{Int})
    classes = sort(unique(y))
    @assert length(classes) == 2 "SVC currently supports only binary classification"
    
    n_samples, n_features = size(X)
    
    # Compute class weights
    if svc.class_weight == :balanced
        class_weights = Dict(c => n_samples / (length(classes) * count(==(c), y)) for c in classes)
    elseif isa(svc.class_weight, Dict)
        class_weights = svc.class_weight
    else
        class_weights = Dict(c => 1.0 for c in classes)
    end
    
    # Prepare labels
    y_binary = 2.0 .* (y .== classes[2]) .- 1.0
    sample_weights = [class_weights[c] for c in y]
    
    # Compute kernel matrix
    K = kernel_function(X, X, svc.kernel, svc.gamma)
    
    # Define the objective function (dual form)
    function objective(alpha)
        return 0.5 * (alpha' * (K .* (y_binary * y_binary')) * alpha) - sum(alpha)
    end
    
    # Define the gradient of the objective function
    function gradient!(G, alpha)
        G .= (K .* (y_binary * y_binary')) * alpha .- 1
    end
    
    # Define constraints
    lower = zeros(n_samples)
    upper = svc.C .* sample_weights
    
    # Solve the optimization problem
    α₀ = fill(1e-6, n_samples)
    result = optimize(
        objective,
        gradient!,
        lower,
        upper,
        α₀,
        Fminbox(LBFGS()),
        Optim.Options(iterations=1000, time_limit=60.0, f_abstol=1e-8)
    )

    # Check convergence
    if !Optim.converged(result)
        @warn "Optimization did not converge. Consider increasing the number of iterations or adjusting parameters."
    end
    
    # Extract support vectors
    alpha = Optim.minimizer(result)
    sv_indices = findall(alpha .> 1e-5)
    svc.support_vectors = X[sv_indices, :]
    svc.dual_coef = alpha[sv_indices] .* y_binary[sv_indices]
    
    # Compute intercept
    svc.intercept = mean(y_binary[sv_indices] .- (K[sv_indices, sv_indices] * (alpha[sv_indices] .* y_binary[sv_indices])))
    
    svc.classes = classes
    return svc
end

function (svc::SVC)(X::Matrix{Float64}; type=nothing)
    if type == :probs
        K = kernel_function(X, svc.support_vectors, svc.kernel, svc.gamma)
        decision_values = K * svc.dual_coef .+ svc.intercept
        probabilities = 1 ./ (1 .+ exp.(-decision_values))
        return hcat(1 .- probabilities, probabilities)
    else
        # decision function
        K = kernel_function(X, svc.support_vectors, svc.kernel, svc.gamma)
        decision_values = K * svc.dual_coef .+ svc.intercept
        return [decision_value > 0 ? svc.classes[2] : svc.classes[1] for decision_value in decision_values]
    end
end