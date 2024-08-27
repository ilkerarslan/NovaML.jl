using LinearAlgebra
using Statistics
using Distances

function silhouette_samples(X::AbstractMatrix, labels::AbstractVector; metric::Union{String, Function, Metric}="euclidean")
    n_samples = size(X, 1)
    n_labels = length(unique(labels))

    if n_labels < 2 || n_labels >= n_samples
        throw(ArgumentError("Number of labels is $n_labels. Valid range is 2 to n_samples - 1 = $(n_samples-1)"))
    end

    # Compute pairwise distances if not precomputed
    distances = if metric == "precomputed"
        X
    elseif metric isa String
        if metric == "euclidean"
            pairwise(Euclidean(), X, dims=1)
        else
            pairwise(eval(Symbol(metric))(), X, dims=1)
        end
    elseif metric isa Function
        [metric(X[i,:], X[j,:]) for i in 1:n_samples, j in 1:n_samples]
    elseif metric isa Metric
        pairwise(metric, X, dims=1)
    else
        throw(ArgumentError("Invalid metric. Use a string, a function, or a Distances.Metric object."))
    end

    silhouette_values = zeros(n_samples)

    for i in 1:n_samples
        # Separate distances to samples in the same cluster and different clusters
        current_cluster = labels[i]
        same_cluster = labels .== current_cluster
        same_cluster[i] = false  # Exclude the sample itself
        other_clusters = unique(labels[labels .!= current_cluster])

        if sum(same_cluster) == 0
            silhouette_values[i] = 0.0
            continue
        end

        # Calculate a: average distance to points in the same cluster
        a = mean(distances[i, same_cluster])

        # Calculate b: minimum average distance to points in different clusters
        b = minimum(
            mean(distances[i, labels .== other_cluster])
            for other_cluster in other_clusters
        )

        silhouette_values[i] = (b - a) / max(a, b)
    end

    return silhouette_values
end