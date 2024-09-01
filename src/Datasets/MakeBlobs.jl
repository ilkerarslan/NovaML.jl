using Random
using LinearAlgebra

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