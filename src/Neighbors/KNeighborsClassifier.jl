using Distances
using Statistics
using StatsBase: sample, Weights
using DataStructures: PriorityQueue, enqueue!, dequeue!, peek

import ...NovaML: AbstractModel

# Custom KDTree implementation
struct KDTreeNode
    point::Vector{Float64}
    index::Int
    left::Union{KDTreeNode, Nothing}
    right::Union{KDTreeNode, Nothing}
    axis::Int
end

struct KDTree
    root::KDTreeNode
    leaf_size::Int
end

mutable struct KNeighborsClassifier <: AbstractModel
    n_neighbors::Int
    weights::Symbol
    algorithm::Symbol
    leaf_size::Int
    metric::Metric
    n_jobs::Union{Int, Nothing}
    tree::Union{KDTree, Nothing}
    X::Union{Matrix{Float64}, Nothing}
    y::Union{Vector, Nothing}
    fitted::Bool

    function KNeighborsClassifier(;
        n_neighbors::Int = 5,
        weights::Symbol = :uniform,
        algorithm::Symbol = :auto,
        leaf_size::Int = 30,
        metric::Metric = Euclidean(),
        n_jobs::Union{Int, Nothing} = nothing
    )
        @assert n_neighbors > 0 "n_neighbors must be positive"
        @assert weights in [:uniform, :distance] "weights must be :uniform or :distance"
        @assert algorithm in [:auto, :kd_tree, :brute] "Invalid algorithm"
        @assert leaf_size > 0 "leaf_size must be positive"

        new(n_neighbors, weights, algorithm, leaf_size, metric, n_jobs, nothing, nothing, nothing, false)
    end
end

# KDTree functions
function build_kdtree(points::Matrix{Float64}, depth::Int, leaf_size::Int)
    n_points, n_dims = size(points)
    if n_points <= leaf_size
        return KDTreeNode(points[1, :], 1, nothing, nothing, 0)
    end

    axis = (depth % n_dims) + 1
    sorted_idx = sortperm(points[:, axis])
    median_idx = div(n_points, 2)

    left_points = points[sorted_idx[1:median_idx], :]
    right_points = points[sorted_idx[median_idx+1:end], :]

    return KDTreeNode(
        points[sorted_idx[median_idx], :],
        sorted_idx[median_idx],
        build_kdtree(left_points, depth + 1, leaf_size),
        build_kdtree(right_points, depth + 1, leaf_size),
        axis
    )
end

function knn_search(tree::KDTree, point::Vector{Float64}, k::Int, metric::Metric)
    pq = PriorityQueue{Tuple{Vector{Float64}, Int}, Float64}()
    search_kdtree!(tree.root, point, k, pq, metric)
    return [item[2] for item in keys(pq)], collect(values(pq))
end

function search_kdtree!(node::KDTreeNode, point::Vector{Float64}, k::Int, pq::PriorityQueue{Tuple{Vector{Float64}, Int}, Float64}, metric::Metric)
    if length(pq) < k || evaluate(metric, point, node.point) < peek(pq)[2]
        if length(pq) == k
            dequeue!(pq)
        end
        enqueue!(pq, (node.point, node.index), -evaluate(metric, point, node.point))
    end

    if node.left === nothing && node.right === nothing
        return
    end

    if point[node.axis] <= node.point[node.axis]
        if node.left !== nothing
            search_kdtree!(node.left, point, k, pq, metric)
        end
        if node.right !== nothing && (length(pq) < k || abs(point[node.axis] - node.point[node.axis]) < peek(pq)[2])
            search_kdtree!(node.right, point, k, pq, metric)
        end
    else
        if node.right !== nothing
            search_kdtree!(node.right, point, k, pq, metric)
        end
        if node.left !== nothing && (length(pq) < k || abs(point[node.axis] - node.point[node.axis]) < peek(pq)[2])
            search_kdtree!(node.left, point, k, pq, metric)
        end
    end
end

# KNeighborsClassifier methods
function (clf::KNeighborsClassifier)(X::Matrix{Float64}, y::Vector)
    clf.X = X
    clf.y = y

    if clf.algorithm == :auto
        clf.algorithm = size(X, 2) < 3 ? :kd_tree : :brute
    end

    if clf.algorithm == :kd_tree
        root = build_kdtree(X, 0, clf.leaf_size)
        clf.tree = KDTree(root, clf.leaf_size)
    end

    clf.fitted = true
    return clf
end

function (clf::KNeighborsClassifier)(X::Matrix{Float64})
    if !clf.fitted
        throw(ErrorException("This KNeighborsClassifier instance is not fitted yet. Call with training data before using it for predictions."))
    end

    predictions = Vector{eltype(clf.y)}(undef, size(X, 1))

    for i in 1:size(X, 1)
        if clf.algorithm == :brute
            distances = [evaluate(clf.metric, X[i, :], clf.X[j, :]) for j in 1:size(clf.X, 1)]
            sorted_indices = sortperm(distances)
            neighbor_indices = sorted_indices[1:clf.n_neighbors]
            neighbor_distances = distances[neighbor_indices]
        else # kd_tree
            neighbor_indices, neighbor_distances = knn_search(clf.tree, X[i, :], clf.n_neighbors, clf.metric)
        end

        if clf.weights == :uniform
            predictions[i] = mode(clf.y[neighbor_indices])
        else # distance weighting
            weights = 1 ./ neighbor_distances
            predictions[i] = mode(sample(clf.y[neighbor_indices], Weights(weights), clf.n_neighbors))
        end
    end

    return predictions
end

# Helper functions
function mode(x)
    if length(x) == 1
        return x[1]
    end
    return argmax(map(y -> count(==(y), x), unique(x)))
end

function Base.show(io::IO, clf::KNeighborsClassifier)
    print(io, "KNeighborsClassifier(n_neighbors=$(clf.n_neighbors), ",
        "weights=$(clf.weights), algorithm=$(clf.algorithm), ",
        "leaf_size=$(clf.leaf_size), metric=$(clf.metric), ",
        "n_jobs=$(clf.n_jobs), fitted=$(clf.fitted))")
end