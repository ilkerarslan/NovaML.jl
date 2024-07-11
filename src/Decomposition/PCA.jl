using LinearAlgebra
import Statistics: mean, cov

export PCA

"""
    PCA

Principal Component Analysis (PCA) implementation.

# Fields
- `n_components::Union{Int, Float64, String}`: Number of components to keep
- `whiten::Bool`: Whether to whiten the output
- `fitted::Bool`: Indicates whether the PCA has been fitted
- `components_::Union{Matrix{Float64}, Nothing}`: Principal components
- `explained_variance_::Union{Vector{Float64}, Nothing}`: Variance explained by each component
- `explained_variance_ratio_::Union{Vector{Float64}, Nothing}`: Ratio of variance explained
- `singular_values_::Union{Vector{Float64}, Nothing}`: Singular values
- `mean_::Union{Vector{Float64}, Nothing}`: Mean of the training data
- `n_samples_::Union{Int, Nothing}`: Number of samples in the training data
- `n_features_::Union{Int, Nothing}`: Number of features in the training data
- `n_components_::Union{Int, Nothing}`: Actual number of components used
- `noise_variance_::Union{Float64, Nothing}`: Estimated noise variance
"""
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

"""
    (pca::PCA)(X::AbstractMatrix{T}, y=nothing) where T <: Real

Fit the PCA model to the input data.

# Arguments
- `X::AbstractMatrix{T}`: Input data matrix
- `y`: Ignored (included for API consistency)

# Returns
- The fitted PCA object
"""
function (pca::PCA)(X::AbstractMatrix{T}, y=nothing) where T <: Real
    if y !== nothing
        @warn "y is ignored in PCA"
    end

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
    return pca
end

"""
    (pca::PCA)(X::AbstractMatrix{T}, mode::Symbol) where T <: Real

Transform or inverse transform data using the fitted PCA model.

# Arguments
- `X::AbstractMatrix{T}`: Input data matrix
- `mode::Symbol`: Either :transform or :inverse_transform

# Returns
- Transformed or inverse transformed data

# Throws
- `ErrorException`: If the PCA hasn't been fitted or if an invalid mode is provided
"""
function (pca::PCA)(X::AbstractMatrix{T}, mode::Symbol) where T <: Real
    if !pca.fitted
        throw(ErrorException("PCA must be fitted before transforming or inverse transforming."))
    end

    if mode == :transform
        X_centered = X .- pca.mean_'
        X_transformed = X_centered * pca.components_

        if pca.whiten
            X_transformed ./= sqrt.(pca.explained_variance_)'
        end

        return X_transformed
    elseif mode == :inverse_transform
        if pca.whiten
            X_unwhitened = X .* sqrt.(pca.explained_variance_)'
        else
            X_unwhitened = X
        end

        return X_unwhitened * pca.components_' .+ pca.mean_'
    else
        throw(ErrorException("Invalid mode. Use :transform or :inverse_transform."))
    end
end

"""
    Base.show(io::IO, pca::PCA)

Custom pretty printing for PCA.

# Arguments
- `io::IO`: The I/O stream
- `pca::PCA`: The PCA object to be printed
"""
function Base.show(io::IO, pca::PCA)
    fitted_status = pca.fitted ? "fitted" : "not fitted"
    n_components = pca.fitted ? pca.n_components_ : pca.n_components
    print(io, "PCA(n_components=$(n_components), whiten=$(pca.whiten), $(fitted_status))")
end