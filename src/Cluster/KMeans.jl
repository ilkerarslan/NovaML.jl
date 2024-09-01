using Random
using LinearAlgebra
using Statistics

import ...NovaML: AbstractModel

"""
    KMeans <: AbstractModel

Represents the K-Means clustering algorithm.

# Fields
- `n_clusters::Int`: The number of clusters to form.
- `init::Union{String, Matrix{Float64}, Function}`: Method for initialization.
- `n_init::Union{Int, String}`: Number of time the k-means algorithm will be run with different centroid seeds.
- `max_iter::Int`: Maximum number of iterations of the k-means algorithm for a single run.
- `tol::Float64`: Relative tolerance with regards to inertia to declare convergence.
- `verbose::Int`: Verbosity mode.
- `random_state::Union{Int, Nothing}`: Determines random number generation for centroid initialization.
- `copy_x::Bool`: When pre-computing distances it is more numerically accurate to center the data first.
- `algorithm::String`: K-means algorithm to use.

# Fitted Attributes
- `cluster_centers_::Union{Matrix{Float64}, Nothing}`: Coordinates of cluster centers.
- `labels_::Union{Vector{Int}, Nothing}`: Labels of each point.
- `inertia_::Union{Float64, Nothing}`: Sum of squared distances of samples to their closest cluster center.
- `n_iter_::Union{Int, Nothing}`: Number of iterations run.
"""
mutable struct KMeans <: AbstractModel
    n_clusters::Int
    init::Union{String, Matrix{Float64}, Function}
    n_init::Union{Int, String}
    max_iter::Int
    tol::Float64
    verbose::Int
    random_state::Union{Int, Nothing}
    copy_x::Bool
    algorithm::String

    # Fitted attributes
    cluster_centers_::Union{Matrix{Float64}, Nothing}
    labels_::Union{Vector{Int}, Nothing}
    inertia_::Union{Float64, Nothing}
    n_iter_::Union{Int, Nothing}
    
    function KMeans(;
        n_clusters::Int=8,
        init::Union{String, Matrix{Float64}, Function}="k-means++",
        n_init::Union{Int, String}="auto",
        max_iter::Int=300,
        tol::Float64=1e-4,
        verbose::Int=0,
        random_state::Union{Int, Nothing}=nothing,
        copy_x::Bool=true,
        algorithm::String="lloyd"
    )
        new(n_clusters, init, n_init, max_iter, tol, verbose, random_state, copy_x, algorithm,
            nothing, nothing, nothing, nothing)
    end
end

"""
    (kmeans::KMeans)(X::AbstractVecOrMat{Float64}, y=nothing; sample_weight=nothing)

Compute k-means clustering.

# Arguments
- `X::AbstractVecOrMat{Float64}`: Training instances to cluster.
- `y`: Ignored. Not used, present for API consistency by convention.
- `sample_weight`: The weights for each observation in X.

# Returns
- If the model is not fitted, returns the fitted model.
- If the model is already fitted, returns the predicted labels for X.
"""
function (kmeans::KMeans)(X::AbstractVecOrMat{Float64}, y=nothing; sample_weight=nothing)
    # If X is a vector, convert it to a matrix
    X_matrix = X isa AbstractVector ? reshape(X, 1, :) : X

    if kmeans.cluster_centers_ === nothing
        # Fitting
        n_samples, n_features = size(X_matrix)
        
        if kmeans.random_state !== nothing
            Random.seed!(kmeans.random_state)
        end
        
        if kmeans.copy_x
            X_matrix = copy(X_matrix)
        end

        best_inertia = Inf
        best_labels = nothing
        best_centers = nothing
        best_n_iter = nothing

        n_init = kmeans.n_init == "auto" ? (kmeans.init == "random" ? 10 : 1) : kmeans.n_init

        for _ in 1:n_init
            centroids = initialize_centroids(kmeans, X_matrix)
            
            labels = Vector{Int}(undef, n_samples)
            n_iter = 0
            for iter in 1:kmeans.max_iter
                n_iter = iter
                old_centroids = copy(centroids)
                
                labels = assign_labels(X_matrix, centroids)
                centroids = update_centroids(X_matrix, labels, kmeans.n_clusters, sample_weight)
                
                if norm(centroids - old_centroids) < kmeans.tol
                    break
                end
            end

            inertia = compute_inertia(X_matrix, centroids, labels, sample_weight)

            if inertia < best_inertia
                best_inertia = inertia
                best_labels = labels
                best_centers = centroids
                best_n_iter = n_iter
            end
        end

        kmeans.cluster_centers_ = best_centers
        kmeans.labels_ = best_labels
        kmeans.inertia_ = best_inertia
        kmeans.n_iter_ = best_n_iter

        return kmeans
    else
        # Predicting
        return assign_labels(X_matrix, kmeans.cluster_centers_)
    end
end

"""
    initialize_centroids(kmeans::KMeans, X::Matrix{Float64})

Initialize the centroids for K-Means clustering.

# Arguments
- `kmeans::KMeans`: The KMeans instance.
- `X::Matrix{Float64}`: The input data.

# Returns
- `Matrix{Float64}`: The initial centroids.
"""
function initialize_centroids(kmeans::KMeans, X::Matrix{Float64})
    n_samples, n_features = size(X)
    
    if typeof(kmeans.init) == String
        if kmeans.init == "k-means++"
            centroids = kmeans_plus_plus(X, kmeans.n_clusters)
        elseif kmeans.init == "random"
            idx = randperm(n_samples)[1:kmeans.n_clusters]
            centroids = X[idx, :]
        else
            error("Unknown initialization method: $(kmeans.init)")
        end
    elseif typeof(kmeans.init) == Matrix{Float64}
        centroids = kmeans.init
    elseif typeof(kmeans.init) <: Function
        centroids = kmeans.init(X, kmeans.n_clusters, kmeans.random_state)
    else
        error("Invalid init parameter type")
    end
    
    return centroids
end

"""
    kmeans_plus_plus(X::Matrix{Float64}, n_clusters::Int)

Perform K-Means++ initialization.

# Arguments
- `X::Matrix{Float64}`: The input data.
- `n_clusters::Int`: The number of clusters.

# Returns
- `Matrix{Float64}`: The initial centroids chosen by K-Means++.
"""
function kmeans_plus_plus(X::Matrix{Float64}, n_clusters::Int)
    n_samples, n_features = size(X)
    centroids = zeros(n_clusters, n_features)
    
    # Choose the first centroid randomly
    centroids[1, :] = X[rand(1:n_samples), :]
    
    # Choose the remaining centroids
    for k in 2:n_clusters
        distances = [minimum([norm(x - centroids[j, :])^2 for j in 1:k-1]) for x in eachrow(X)]
        probabilities = distances ./ sum(distances)
        cumulative_probabilities = cumsum(probabilities)
        r = rand()
        for (i, p) in enumerate(cumulative_probabilities)
            if r <= p
                centroids[k, :] = X[i, :]
                break
            end
        end
    end
    
    return centroids
end

"""
    assign_labels(X::AbstractMatrix{Float64}, centroids::Matrix{Float64})

Assign labels to data points based on the nearest centroid.

# Arguments
- `X::AbstractMatrix{Float64}`: The input data.
- `centroids::Matrix{Float64}`: The current centroids.

# Returns
- `Vector{Int}`: The assigned labels for each data point.
"""
function assign_labels(X::AbstractMatrix{Float64}, centroids::Matrix{Float64})
    return [argmin([norm(x - c)^2 for c in eachrow(centroids)]) for x in eachrow(X)]
end

"""
    update_centroids(X::Matrix{Float64}, labels::Vector{Int}, n_clusters::Int, sample_weight=nothing)

Update the centroids based on the current label assignments.

# Arguments
- `X::Matrix{Float64}`: The input data.
- `labels::Vector{Int}`: The current label assignments.
- `n_clusters::Int`: The number of clusters.
- `sample_weight`: The weights for each observation in X.

# Returns
- `Matrix{Float64}`: The updated centroids.
"""
function update_centroids(X::Matrix{Float64}, labels::Vector{Int}, n_clusters::Int, sample_weight=nothing)
    n_samples, n_features = size(X)
    centroids = zeros(n_clusters, n_features)
    
    if sample_weight === nothing
        sample_weight = ones(n_samples)
    end
    
    for k in 1:n_clusters
        mask = labels .== k
        if any(mask)
            centroids[k, :] = sum(X[mask, :] .* sample_weight[mask], dims=1) ./ sum(sample_weight[mask])
        else
            centroids[k, :] = X[rand(1:n_samples), :]
        end
    end
    
    return centroids
end

"""
    compute_inertia(X::Matrix{Float64}, centroids::Matrix{Float64}, labels::Vector{Int}, sample_weight=nothing)

Compute the inertia, the sum of squared distances of samples to their closest cluster center.

# Arguments
- `X::Matrix{Float64}`: The input data.
- `centroids::Matrix{Float64}`: The current centroids.
- `labels::Vector{Int}`: The current label assignments.
- `sample_weight`: The weights for each observation in X.

# Returns
- `Float64`: The computed inertia.
"""
function compute_inertia(X::Matrix{Float64}, centroids::Matrix{Float64}, labels::Vector{Int}, sample_weight=nothing)
    if sample_weight === nothing
        sample_weight = ones(size(X, 1))
    end
    return sum(sample_weight .* [norm(X[i, :] - centroids[labels[i], :])^2 for i in 1:size(X, 1)])
end

"""
    get_params(kmeans::KMeans; deep=true)

Get parameters for this estimator.

# Arguments
- `kmeans::KMeans`: The KMeans instance.
- `deep::Bool`: If True, will return the parameters for this estimator and contained subobjects that are estimators.

# Returns
- `Dict`: Parameter names mapped to their values.
"""
function get_params(kmeans::KMeans; deep=true)
    return Dict(
        "n_clusters" => kmeans.n_clusters,
        "init" => kmeans.init,
        "n_init" => kmeans.n_init,
        "max_iter" => kmeans.max_iter,
        "tol" => kmeans.tol,
        "verbose" => kmeans.verbose,
        "random_state" => kmeans.random_state,
        "copy_x" => kmeans.copy_x,
        "algorithm" => kmeans.algorithm
    )
end

"""
    set_params!(kmeans::KMeans; params...)

Set the parameters of this estimator.

# Arguments
- `kmeans::KMeans`: The KMeans instance.
- `params...`: Estimator parameters.

# Returns
- `KMeans`: The estimator instance.
"""
function set_params!(kmeans::KMeans; params...)
    for (key, value) in params
        setproperty!(kmeans, Symbol(key), value)
    end
    return kmeans
end

"""
    fit_predict(kmeans::KMeans, X::Matrix{Float64}, y=nothing; sample_weight=nothing)

Compute cluster centers and predict cluster index for each sample.

# Arguments
- `kmeans::KMeans`: The KMeans instance.
- `X::Matrix{Float64}`: New data to transform.
- `y`: Ignored.
- `sample_weight`: The weights for each observation in X.

# Returns
- `Vector{Int}`: Index of the cluster each sample belongs to.
"""
function fit_predict(kmeans::KMeans, X::Matrix{Float64}, y=nothing; sample_weight=nothing)
    kmeans(X, y; sample_weight=sample_weight)
    return kmeans.labels_
end

"""
    fit_transform(kmeans::KMeans, X::Matrix{Float64}, y=nothing; sample_weight=nothing)

Compute clustering and transform X to cluster-distance space.

# Arguments
- `kmeans::KMeans`: The KMeans instance.
- `X::Matrix{Float64}`: New data to transform.
- `y`: Ignored.
- `sample_weight`: The weights for each observation in X.

# Returns
- `Matrix{Float64}`: X transformed in the new space.
"""
function fit_transform(kmeans::KMeans, X::Matrix{Float64}, y=nothing; sample_weight=nothing)
    kmeans(X, y; sample_weight=sample_weight)
    return transform(kmeans, X)
end

"""
    transform(kmeans::KMeans, X::Matrix{Float64})

Transform X to a cluster-distance space.

# Arguments
- `kmeans::KMeans`: The KMeans instance.
- `X::Matrix{Float64}`: New data to transform.

# Returns
- `Matrix{Float64}`: X transformed in the new space.
"""
function transform(kmeans::KMeans, X::Matrix{Float64})
    if kmeans.cluster_centers_ === nothing
        error("KMeans model is not fitted yet. Call the model with training data first.")
    end
    return [norm(x - c)^2 for x in eachrow(X), c in eachrow(kmeans.cluster_centers_)]
end

"""
    score(kmeans::KMeans, X::Matrix{Float64}, y=nothing; sample_weight=nothing)

Opposite of the value of X on the K-means objective.

# Arguments
- `kmeans::KMeans`: The KMeans instance.
- `X::Matrix{Float64}`: New data.
- `y`: Ignored.
- `sample_weight`: The weights for each observation in X.

# Returns
- `Float64`: Opposite of the value of X on the K-means objective.
"""
function score(kmeans::KMeans, X::Matrix{Float64}, y=nothing; sample_weight=nothing)
    if kmeans.cluster_centers_ === nothing
        error("KMeans model is not fitted yet. Call the model with training data first.")
    end
    labels = kmeans(X)
    return -compute_inertia(X, kmeans.cluster_centers_, labels, sample_weight)
end

"""
    Base.show(io::IO, kmeans::KMeans)

Custom show method for KMeans instances.

# Arguments
- `io::IO`: The I/O stream.
- `kmeans::KMeans`: The KMeans instance to display.
"""
function Base.show(io::IO, kmeans::KMeans)
    fitted_status = kmeans.cluster_centers_ === nothing ? "unfitted" : "fitted"
    print(io, "KMeans(n_clusters=$(kmeans.n_clusters), $(fitted_status))")
end