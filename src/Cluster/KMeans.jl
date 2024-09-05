using Random
using LinearAlgebra
using Statistics

import ...NovaML: AbstractModel

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

function (kmeans::KMeans)(X::AbstractVecOrMat{Float64}, y=nothing; sample_weight=nothing)
    X_matrix = X isa AbstractVector ? reshape(X, 1, :) : X

    if kmeans.cluster_centers_ === nothing
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
        return assign_labels(X_matrix, kmeans.cluster_centers_)
    end
end

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

function kmeans_plus_plus(X::Matrix{Float64}, n_clusters::Int)
    n_samples, n_features = size(X)
    centroids = zeros(n_clusters, n_features)
    
    centroids[1, :] = X[rand(1:n_samples), :]
    
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

function assign_labels(X::AbstractMatrix{Float64}, centroids::Matrix{Float64})
    return [argmin([norm(x - c)^2 for c in eachrow(centroids)]) for x in eachrow(X)]
end

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

function compute_inertia(X::Matrix{Float64}, centroids::Matrix{Float64}, labels::Vector{Int}, sample_weight=nothing)
    if sample_weight === nothing
        sample_weight = ones(size(X, 1))
    end
    return sum(sample_weight .* [norm(X[i, :] - centroids[labels[i], :])^2 for i in 1:size(X, 1)])
end

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

function set_params!(kmeans::KMeans; params...)
    for (key, value) in params
        setproperty!(kmeans, Symbol(key), value)
    end
    return kmeans
end

function fit_predict(kmeans::KMeans, X::Matrix{Float64}, y=nothing; sample_weight=nothing)
    kmeans(X, y; sample_weight=sample_weight)
    return kmeans.labels_
end

function fit_transform(kmeans::KMeans, X::Matrix{Float64}, y=nothing; sample_weight=nothing)
    kmeans(X, y; sample_weight=sample_weight)
    return transform(kmeans, X)
end

function transform(kmeans::KMeans, X::Matrix{Float64})
    if kmeans.cluster_centers_ === nothing
        error("KMeans model is not fitted yet. Call the model with training data first.")
    end
    return [norm(x - c)^2 for x in eachrow(X), c in eachrow(kmeans.cluster_centers_)]
end

function score(kmeans::KMeans, X::Matrix{Float64}, y=nothing; sample_weight=nothing)
    if kmeans.cluster_centers_ === nothing
        error("KMeans model is not fitted yet. Call the model with training data first.")
    end
    labels = kmeans(X)
    return -compute_inertia(X, kmeans.cluster_centers_, labels, sample_weight)
end

function Base.show(io::IO, kmeans::KMeans)
    fitted_status = kmeans.cluster_centers_ === nothing ? "unfitted" : "fitted"
    print(io, "KMeans(n_clusters=$(kmeans.n_clusters), $(fitted_status))")
end