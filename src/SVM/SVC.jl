
using LinearAlgebra
using Statistics
using Distances

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

function kernel_matrix!(K::Matrix{Float64}, X::Matrix{Float64}, Y::Matrix{Float64}, kernel::Symbol, gamma::Union{Float64,Symbol})
    if kernel == :linear
        mul!(K, X, Y')
    elseif kernel == :rbf
        gamma_value = (gamma == :scale) ? 1.0 / size(X, 2) : gamma
        pairwise!(K, SqEuclidean(), X', Y')
        K .*= -gamma_value
        @. K = exp(K)
    else
        error("Unsupported kernel: $kernel")
    end
end

function smo_optimize!(alpha::Vector{Float64}, y::Vector{Float64}, K::Matrix{Float64}, C::Float64, tol::Float64, max_passes::Int)
    n = length(y)
    passes = 0
    while passes < max_passes
        num_changed_alphas = 0
        for i in 1:n
            Ei = sum(alpha .* y .* K[:, i]) - y[i]
            if (y[i] * Ei < -tol && alpha[i] < C) || (y[i] * Ei > tol && alpha[i] > 0)
                j = rand(1:n)
                while j == i
                    j = rand(1:n)
                end
                Ej = sum(alpha .* y .* K[:, j]) - y[j]
                
                alpha_i_old, alpha_j_old = alpha[i], alpha[j]
                
                if y[i] != y[j]
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                end
                
                if L == H
                    continue
                end
                
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0
                    continue
                end
                
                alpha[j] = alpha[j] - (y[j] * (Ei - Ej)) / eta
                alpha[j] = clamp(alpha[j], L, H)
                
                if abs(alpha[j] - alpha_j_old) < 1e-5
                    continue
                end
                
                alpha[i] = alpha[i] + y[i] * y[j] * (alpha_j_old - alpha[j])
                
                num_changed_alphas += 1
            end
        end
        
        if num_changed_alphas == 0
            passes += 1
        else
            passes = 0
        end
    end
end

function (svc::SVC)(X::Matrix{Float64}, y::AbstractVector)
    classes = sort(unique(y))
    @assert length(classes) == 2 "SVC currently supports only binary classification"
    
    n_samples, n_features = size(X)
    
    # Convert BitVector to Vector{Int} if necessary
    y_int = y isa BitVector ? Vector{Int}(y) : y
    
    # Compute class weights
    if svc.class_weight == :balanced
        class_weights = Dict(c => n_samples / (length(classes) * count(==(c), y_int)) for c in classes)
    elseif isa(svc.class_weight, Dict)
        class_weights = svc.class_weight
    else
        class_weights = Dict(c => 1.0 for c in classes)
    end
    
    # Prepare labels
    y_binary = 2.0 .* (y_int .== classes[2]) .- 1.0
    sample_weights = [class_weights[c] for c in y_int]
    
    # Compute kernel matrix
    K = Matrix{Float64}(undef, n_samples, n_samples)
    kernel_matrix!(K, X, X, svc.kernel, svc.gamma)
    
    # Initialize alpha
    alpha = zeros(n_samples)
    
    # Optimize using SMO algorithm
    smo_optimize!(alpha, y_binary, K, svc.C, 1e-3, 100)
    
    # Extract support vectors
    sv_indices = findall(alpha .> 1e-5)
    svc.support_vectors = X[sv_indices, :]
    svc.dual_coef = alpha[sv_indices] .* y_binary[sv_indices]
    
    # Compute intercept
    svc.intercept = mean(y_binary[sv_indices] .- (K[sv_indices, sv_indices] * (alpha[sv_indices] .* y_binary[sv_indices])))
    
    svc.classes = classes
    return svc
end


function (svc::SVC)(X::Matrix{Float64}; type=nothing)
    K = Matrix{Float64}(undef, size(X, 1), size(svc.support_vectors, 1))
    kernel_matrix!(K, X, svc.support_vectors, svc.kernel, svc.gamma)
    decision_values = K * svc.dual_coef .+ svc.intercept
    
    if type == :probs
        probabilities = 1 ./ (1 .+ exp.(-decision_values))
        return hcat(1 .- probabilities, probabilities)
    else
        return [decision_value > 0 ? svc.classes[2] : svc.classes[1] for decision_value in decision_values]
    end
end