using Random
using LinearAlgebra

"""
    make_blobs(;
        n_samples::Union{Int, Vector{Int}} = 100,
        n_features::Int = 2,
        centers::Union{Int, Matrix{Float64}} = nothing,
        cluster_std::Union{Float64, Vector{Float64}} = 1.0,
        center_box::Tuple{Float64, Float64} = (-10.0, 10.0),
        shuffle::Bool = true,
        random_state::Union{Int, Nothing} = nothing,
        return_centers::Bool = false
    )

Generate isotropic Gaussian blobs for clustering.

# Arguments
- `n_samples::Union{Int, Vector{Int}}`: The total number of points equally divided among clusters, or the number of samples per cluster.
- `n_features::Int`: The number of features for each sample.
- `centers::Union{Int, Matrix{Float64}}`: The number of centers to generate, or a matrix of center locations.
- `cluster_std::Union{Float64, Vector{Float64}}`: The standard deviation of the clusters.
- `center_box::Tuple{Float64, Float64}`: The bounding box for each cluster center when centers are generated at random.
- `shuffle::Bool`: Shuffle the samples.
- `random_state::Union{Int, Nothing}`: Determines random number generation for dataset creation.
- `return_centers::Bool`: If true, returns the centers in addition to X and y.

# Returns
- If `return_centers` is false:
    - `X::Matrix{Float64}`: Generated samples.
    - `y::Vector{Int}`: The integer labels for cluster membership of each sample.
- If `return_centers` is true:
    - `X::Matrix{Float64}`: Generated samples.
    - `y::Vector{Int}`: The integer labels for cluster membership of each sample.
    - `centers::Matrix{Float64}`: The centers used to generate the data.

# Description
This function generates samples from isotropic Gaussian blobs for clustering.
It can be used for testing clustering algorithms or as a simple dataset for demonstration purposes.

# Example
```julia
# Generate a simple dataset with 3 clusters
X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

# Generate a dataset with specified centers and return the centers
centers = [0 0; 1 1; 2 2]
X, y, centers = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, return_centers=true)

# Notes
If centers is an int, it is interpreted as the number of centers to generate, and they are generated randomly within center_box.
- If centers is a 2-d array, it is interpreted as the actual centers to use, and n_features is ignored in this case.
- If n_samples is an int, it is interpreted as the total number of samples, which are then evenly divided among clusters.
- If n_samples is an array, it is interpreted as the number of samples per cluster.
"""
function make_blobs(;
    n_samples::Union{Int, Vector{Int}} = 100,
    n_features::Int = 2,
    centers::Union{Int, Matrix{Float64}} = nothing,
    cluster_std::Union{Float64, Vector{Float64}} = 1.0,
    center_box::Tuple{Float64, Float64} = (-10.0, 10.0),
    shuffle::Bool = true,
    random_state::Union{Int, Nothing} = nothing,
    return_centers::Bool = false)

    rng = Random.MersenneTwister(random_state)

    if centers === nothing
        centers = 3
    end

    if isa(n_samples, Int)
        if isa(centers, Int)
            n_centers = centers
            centers = rand(rng, n_centers, n_features) .* (center_box[2] - center_box[1]) .+ center_box[1]
        else
            n_centers = size(centers, 1)
        end
        n_samples_per_center = fill(n_samples ÷ n_centers, n_centers)
        n_samples_per_center[1:n_samples % n_centers] .+= 1
    else
        n_centers = length(n_samples)
        n_samples_per_center = n_samples
        if centers === nothing
            centers = rand(rng, n_centers, n_features) .* (center_box[2] - center_box[1]) .+ center_box[1]
        end
    end

    if isa(cluster_std, Number)
        cluster_std = fill(cluster_std, n_centers)
    end

    X = zeros(sum(n_samples_per_center), n_features)
    y = zeros(Int, sum(n_samples_per_center))

    start = 1
    for (i, n) in enumerate(n_samples_per_center)
        stop = start + n - 1
        X[start:stop, :] = randn(rng, n, n_features) .* cluster_std[i] .+ centers[i:i, :]
        y[start:stop] .= i - 1
        start = stop + 1
    end

    if shuffle
        idx = Random.shuffle(rng, 1:size(X, 1))
        X = X[idx, :]
        y = y[idx]
    end

    if return_centers
        return X, y, centers
    else
        return X, y
    end
end