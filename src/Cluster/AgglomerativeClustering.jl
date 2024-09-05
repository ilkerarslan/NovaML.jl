mutable struct AgglomerativeClustering
    n_clusters::Union{Int, Nothing}
    metric::Union{String, Function}
    memory::Union{String, Nothing}
    connectivity::Union{AbstractMatrix, Function, Nothing}
    compute_full_tree::Union{Bool, String}
    linkage::String
    distance_threshold::Union{Float64, Nothing}
    compute_distances::Bool
    
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

function (clustering::AgglomerativeClustering)(X::AbstractMatrix; y=nothing)
    n_samples, n_features = size(X)
    
    distances = compute_distances(X, clustering.metric)
    
    clusters = collect(1:n_samples)
    
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
        
        if clustering.distance_threshold !== nothing && min_dist > clustering.distance_threshold
            break
        end        
        
        old_cluster = clusters[min_j]
        new_cluster = clusters[min_i]
        for i in 1:n_samples
            if clusters[i] == old_cluster
                clusters[i] = new_cluster
            end
        end
        
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
    
    unique_clusters = unique(clusters)
    clustering.labels_ = [findfirst(==(c), unique_clusters) for c in clusters]
    
    clustering.n_leaves_ = n_samples
    clustering.n_connected_components_ = length(unique_clusters)
    
    return clustering
end

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

function (clustering::AgglomerativeClustering)(X::AbstractMatrix, type::Symbol)
    if type == :fit_predict
        clustering(X)
        return clustering.labels_
    else
        error("Unsupported type: $type")
    end
end