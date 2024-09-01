"""
    AgglomerativeClustering

A struct representing Agglomerative Clustering, a hierarchical clustering algorithm.

# Fields
- `n_clusters::Union{Int, Nothing}`: The number of clusters to find. If `nothing`, it must be used with `distance_threshold`.
- `metric::Union{String, Function}`: The metric to use for distance computation. Can be "euclidean", "manhattan", or a custom function.
- `memory::Union{String, Nothing}`: Used to cache the distance matrix between iterations.
- `connectivity::Union{AbstractMatrix, Function, Nothing}`: Connectivity matrix or callable to be used.
- `compute_full_tree::Union{Bool, String}`: Whether to compute the full tree or stop early.
- `linkage::String`: The linkage criterion to use. Can be "ward", "complete", "average", or "single".
- `distance_threshold::Union{Float64, Nothing}`: The threshold to stop clustering.
- `compute_distances::Bool`: Whether to compute distances.

# Fitted Attributes
- `labels_::Vector{Int}`: Cluster labels for each point.
- `n_leaves_::Int`: Number of leaves in the hierarchical tree.
- `n_connected_components_::Int`: Number of connected components in the graph.
- `children_::Matrix{Int}`: The children of each non-leaf node.
- `distances_::Vector{Float64}`: Distances between nodes in the tree.

# Constructor
    AgglomerativeClustering(;
        n_clusters::Union{Int, Nothing}=2,
        metric::Union{String, Function}="euclidean",
        memory::Union{String, Nothing}=nothing,
        connectivity::Union{AbstractMatrix, Function, Nothing}=nothing,
        compute_full_tree::Union{Bool, String}="auto",
        linkage::String="ward",
        distance_threshold::Union{Float64, Nothing}=nothing,
        compute_distances::Bool=false
    )

Constructs an AgglomerativeClustering object with the specified parameters.

# Examples
```julia
# Create an AgglomerativeClustering object with 3 clusters
clustering = AgglomerativeClustering(n_clusters=3)

# Create an AgglomerativeClustering object with a distance threshold
clustering = AgglomerativeClustering(distance_threshold=1.5, linkage="single")
```
"""
mutable struct AgglomerativeClustering
    n_clusters::Union{Int, Nothing}
    metric::Union{String, Function}
    memory::Union{String, Nothing}
    connectivity::Union{AbstractMatrix, Function, Nothing}
    compute_full_tree::Union{Bool, String}
    linkage::String
    distance_threshold::Union{Float64, Nothing}
    compute_distances::Bool
    
    # Fitted attributes
    labels_::Vector{Int}
    n_leaves_::Int
    n_connected_components_::Int
    children_::Matrix{Int}
    distances_::Vector{Float64}

    function AgglomerativeClustering(;
        n_clusters::Union{Int, Nothing}=2,
        metric::Union{String, Function}="euclidean",
        memory::Union{String, Nothing}=nothing,
        connectivity::Union{AbstractMatrix, Function, Nothing}=nothing,
        compute_full_tree::Union{Bool, String}="auto",
        linkage::String="ward",
        distance_threshold::Union{Float64, Nothing}=nothing,
        compute_distances::Bool=false
    )
        if n_clusters === nothing && distance_threshold === nothing
            error("Either n_clusters or distance_threshold must be specified")
        end
        if n_clusters !== nothing && distance_threshold !== nothing
            error("n_clusters and distance_threshold cannot be specified simultaneously")
        end
        if linkage == "ward" && metric != "euclidean"
            error("Ward linkage only supports Euclidean distance")
        end
        new(n_clusters, metric, memory, connectivity, compute_full_tree, linkage,
            distance_threshold, compute_distances, Int[], 0, 0, Matrix{Int}(undef, 0, 0), Float64[])
    end
end

"""
    (clustering::AgglomerativeClustering)(X::AbstractMatrix; y=nothing)
Perform agglomerative clustering on the input data.

# Arguments

X::AbstractMatrix: The input data matrix where each row is a sample and each column is a feature.
y=nothing: Ignored. Present for API consistency.

# Returns

clustering::AgglomerativeClustering: The fitted clustering object.

#Examples
```julia
X = rand(100, 5)  # 100 samples, 5 features
clustering = AgglomerativeClustering(n_clusters=3)
fitted_clustering = clustering(X)
```
"""
function (clustering::AgglomerativeClustering)(X::AbstractMatrix; y=nothing)
    n_samples, n_features = size(X)
    
    # Compute distance matrix
    distances = compute_distances(X, clustering.metric)
    
    # Initialize each point as a cluster
    clusters = collect(1:n_samples)
    
    # Main clustering loop
    while length(unique(clusters)) > clustering.n_clusters
        # Find the two closest clusters
        min_dist = Inf
        min_i, min_j = 0, 0
        for i in 1:n_samples
            for j in i+1:n_samples
                if distances[i, j] < min_dist && clusters[i] != clusters[j]
                    min_dist = distances[i, j]
                    min_i, min_j = i, j
                end
            end
        end
        
        # Check distance threshold
        if clustering.distance_threshold !== nothing && min_dist > clustering.distance_threshold
            break
        end
        
        # Merge clusters
        old_cluster = clusters[min_j]
        new_cluster = clusters[min_i]
        for i in 1:n_samples
            if clusters[i] == old_cluster
                clusters[i] = new_cluster
            end
        end
        
        # Update distances
        for k in 1:n_samples
            if k != min_i && k != min_j
                new_dist = if clustering.linkage == "single"
                    min(distances[min_i, k], distances[min_j, k])
                elseif clustering.linkage == "complete"
                    max(distances[min_i, k], distances[min_j, k])
                elseif clustering.linkage == "average"
                    (distances[min_i, k] + distances[min_j, k]) / 2
                elseif clustering.linkage == "ward"
                    error("Ward linkage not implemented in this version")
                end
                distances[min_i, k] = distances[k, min_i] = new_dist
            end
        end
        distances[min_j, :] .= Inf
        distances[:, min_j] .= Inf
    end
    
    # Assign cluster labels
    unique_clusters = unique(clusters)
    clustering.labels_ = [findfirst(==(c), unique_clusters) for c in clusters]
    
    # Compute other attributes
    clustering.n_leaves_ = n_samples
    clustering.n_connected_components_ = length(unique_clusters)
    
    return clustering
end

"""
    compute_distances(X::AbstractMatrix, metric::Union{String, Function})
Compute the distance matrix for the input data using the specified metric.

# Arguments

X::AbstractMatrix: The input data matrix.
metric::Union{String, Function}: The distance metric to use. Can be "euclidean", "manhattan", or a custom function.

# Returns

distances::Matrix: The computed distance matrix.

# Examples
```julia
X = rand(10, 3)
distances = compute_distances(X, "euclidean")
```
"""
function compute_distances(X::AbstractMatrix, metric::Union{String, Function})
    n_samples = size(X, 1)
    distances = zeros(n_samples, n_samples)
    
    for i in 1:n_samples
        for j in i+1:n_samples
            if isa(metric, String)
                if metric == "euclidean"
                    distances[i, j] = sqrt(sum((X[i, :] .- X[j, :]).^2))
                elseif metric == "manhattan"
                    distances[i, j] = sum(abs.(X[i, :] .- X[j, :]))
                else
                    error("Unsupported metric: $metric")
                end
            else
                distances[i, j] = metric(X[i, :], X[j, :])
            end
            distances[j, i] = distances[i, j]
        end
    end
    
    return distances
end

"""
(clustering::AgglomerativeClustering)(X::AbstractMatrix, type::Symbol)
Fit the clustering model and return the cluster labels.

# Arguments

X::AbstractMatrix: The input data matrix.
type::Symbol: Must be :fit_predict to fit the model and return labels.

# Returns

labels::Vector{Int}: The cluster labels for each input sample.

# Examples
```julia
X = rand(100, 5)
clustering = AgglomerativeClustering(n_clusters=3)
labels = clustering(X, :fit_predict)
```
"""
function (clustering::AgglomerativeClustering)(X::AbstractMatrix, type::Symbol)
    if type == :fit_predict
        clustering(X)
        return clustering.labels_
    else
        error("Unsupported type: $type")
    end
end