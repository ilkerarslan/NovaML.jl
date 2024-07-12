using LinearAlgebra
import Statistics: mean, cov

export PCA

mutable struct PCA
    n_components::Union{Int, Float64, String, Nothing}
    whiten::Bool
    fitted::Bool
    components_::Union{Matrix{Float64}, Nothing}
    explained_variance_::Union{Vector{Float64}, Nothing}
    explained_variance_ratio_::Union{Vector{Float64}, Nothing}
    singular_values_::Union{Vector{Float64}, Nothing}
    mean_::Union{Vector{Float64}, Nothing}
    n_samples_::Union{Int, Nothing}
    n_features_::Union{Int, Nothing}
    n_components_::Union{Int, Nothing}
    noise_variance_::Union{Float64, Nothing}

    function PCA(; n_components=nothing, whiten=false)
        new(n_components, whiten, false, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    end
end


function (pca::PCA)(X::AbstractMatrix{T}) where T <: Real
    if !pca.fitted
        n_samples, n_features = size(X)
        pca.n_samples_ = n_samples
        pca.n_features_ = n_features

        # Center the data
        pca.mean_ = mean(X, dims=1)[1,:]
        X_centered = X .- pca.mean_'

        # Perform SVD
        U, S, Vt = svd(X_centered)

        # Determine number of components
        if pca.n_components === nothing
            pca.n_components_ = min(n_samples, n_features)
        elseif typeof(pca.n_components) <: Int
            pca.n_components_ = min(pca.n_components, min(n_samples, n_features))
        elseif typeof(pca.n_components) <: Float64
            cumulative_variance_ratio = cumsum(S.^2) / sum(S.^2)
            pca.n_components_ = findfirst(cumulative_variance_ratio .>= pca.n_components)
        elseif pca.n_components == "mle"
            # Implement MLE method here if needed
            error("MLE method not implemented yet")
        else
            error("Invalid n_components parameter")
        end

        # Store results
        pca.components_ = Vt[1:pca.n_components_, :]'
        pca.explained_variance_ = (S.^2) ./ (n_samples - 1)
        pca.explained_variance_ratio_ = pca.explained_variance_ ./ sum(pca.explained_variance_)
        pca.singular_values_ = S[1:pca.n_components_]

        if pca.n_components_ < min(n_samples, n_features)
            pca.noise_variance_ = mean(pca.explained_variance_[pca.n_components_+1:end])
        else
            pca.noise_variance_ = 0.0
        end

        pca.fitted = true        
    end
    X_centered = X .- pca.mean_'
    X_transformed = X_centered * pca.components_

    if pca.whiten
        X_transformed ./= sqrt.(pca.explained_variance_)'
    end

    return X_transformed
end


function (pca::PCA)(X::AbstractMatrix{T}, mode::Symbol) where T <: Real    

    if mode == :inverse_transform
        if pca.whiten
            X_unwhitened = X .* sqrt.(pca.explained_variance_)'
        else
            X_unwhitened = X
        end

        return X_unwhitened * pca.components_' .+ pca.mean_'
    else
        throw(ErrorException("Mode can only be :inverse_transform."))
    end
end


function Base.show(io::IO, pca::PCA)
    fitted_status = pca.fitted ? "fitted" : "not fitted"
    n_components = pca.fitted ? pca.n_components_ : pca.n_components
    print(io, "PCA(n_components=$(n_components), whiten=$(pca.whiten), $(fitted_status))")
end