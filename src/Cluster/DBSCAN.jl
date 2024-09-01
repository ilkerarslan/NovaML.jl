using LinearAlgebra
using SparseArrays
using ...NovaML.Neighbors: KNeighborsClassifier
using Distances

"""
    DBSCAN

A struct representing the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) clustering algorithm.

# Fields
- `eps::Float64`: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
- `min_samples::Int`: The number of samples in a neighborhood for a point to be considered as a core point.
- `metric::Union{String, Metric}`: The metric to use when calculating distance between instances.
- `metric_params::Union{Nothing, Dict}`: Additional keyword arguments for the metric function.
- `algorithm::Symbol`: The algorithm to be used by the NearestNeighbors module.
- `leaf_size::Int`: Leaf size passed to BallTree or KDTree.
- `p::Union{Nothing, Float64}`: The power of the Minkowski metric to be used to calculate distance between points.
- `n_jobs::Union{Nothing, Int}`: The number of parallel jobs to run.

# Fitted Attributes
- `core_sample_indices_::Vector{Int}`: Indices of core samples.
- `components_::Matrix{Float64}`: Copy of each core sample found by training.
- `labels_::Vector{Int}`: Cluster labels for each point in the dataset given to fit().
- `n_features_in_::Int`: Number of features seen during fit.
- `feature_names_in_::Vector{String}`: Names of features seen during fit.
- `fitted::Bool`: Whether the model has been fitted.

# Constructor
    DBSCAN(;
        eps::Float64 = 0.5,
        min_samples::Int = 5,
        metric::Union{String, Metric} = "euclidean",
        metric_params::Union{Nothing, Dict} = nothing,
        algorithm::Symbol = :auto,
        leaf_size::Int = 30,
        p::Union{Nothing, Float64} = nothing,
        n_jobs::Union{Nothing, Int} = nothing
    )

Constructs a DBSCAN object with the specified parameters.

# Examples
```julia
# Create a DBSCAN object with default parameters
dbscan = DBSCAN()

# Create a DBSCAN object with custom parameters
dbscan = DBSCAN(eps=0.7, min_samples=10, metric="manhattan")
"""
mutable struct DBSCAN
    eps::Float64
    min_samples::Int
    metric::Union{String, Metric}
    metric_params::Union{Nothing, Dict}
    algorithm::Symbol
    leaf_size::Int
    p::Union{Nothing, Float64}
    n_jobs::Union{Nothing, Int}
    
    # Fitted attributes
    core_sample_indices_::Vector{Int}
    components_::Matrix{Float64}
    labels_::Vector{Int}
    n_features_in_::Int
    feature_names_in_::Vector{String}
    fitted::Bool

    function DBSCAN(;
        eps::Float64 = 0.5,
        min_samples::Int = 5,
        metric::Union{String, Metric} = "euclidean",
        metric_params::Union{Nothing, Dict} = nothing,
        algorithm::Symbol = :auto,
        leaf_size::Int = 30,
        p::Union{Nothing, Float64} = nothing,
        n_jobs::Union{Nothing, Int} = nothing
    )
        new(eps, min_samples, metric, metric_params, algorithm, leaf_size, p, n_jobs, 
            Int[], Matrix{Float64}(undef, 0, 0), Int[], 0, String[], false)
    end
end

"""
    (dbscan::DBSCAN)(X::AbstractMatrix, y=nothing; sample_weight=nothing)
Perform DBSCAN clustering on the input data.

# Arguments
X::AbstractMatrix: The input data matrix where each row is a sample and each column is a feature.
y=nothing: Ignored. Present for API consistency.
sample_weight=nothing: Weight of each sample, used in computing the number of neighbors within eps.

# Returns
dbscan::DBSCAN: The fitted DBSCAN object.

# Examples
```julia
X = rand(100, 5)  # 100 samples, 5 features
dbscan = DBSCAN(eps=0.5, min_samples=5)
fitted_dbscan = dbscan(X)
```
"""
function (dbscan::DBSCAN)(X::AbstractMatrix, y=nothing; sample_weight=nothing)
    if !dbscan.fitted
        # Fitting the model
        n_samples, n_features = size(X)
        dbscan.n_features_in_ = n_features
        
        if sample_weight === nothing
            sample_weight = ones(n_samples)
        end
        
        # Create distance matrix
        dist_metric = if typeof(dbscan.metric) == String
            if dbscan.metric == "euclidean"
                Euclidean()
            elseif dbscan.metric == "manhattan"
                Cityblock()
            elseif dbscan.metric == "minkowski"
                Minkowski(dbscan.p === nothing ? 2 : dbscan.p)
            else
                error("Unsupported metric: $(dbscan.metric)")
            end
        else
            dbscan.metric
        end
        
        dist_matrix = pairwise(dist_metric, X', X')
        
        # Find neighbors within eps
        neighbors = [findall(x -> x <= dbscan.eps, dist_matrix[i, :]) for i in 1:n_samples]
        
        # Initialize labels and core samples
        labels = fill(-1, n_samples)
        core_samples = falses(n_samples)
        
        # Find core samples
        for i in 1:n_samples
            if sum(sample_weight[neighbors[i]]) >= dbscan.min_samples
                core_samples[i] = true
            end
        end
        
        # Assign cluster labels
        cluster_label = 0
        for i in 1:n_samples
            if labels[i] != -1 || !core_samples[i]
                continue
            end
            
            cluster_label += 1
            labels[i] = cluster_label
            
            # Expand cluster
            stack = neighbors[i]
            while !isempty(stack)
                neighbor = pop!(stack)
                if labels[neighbor] == -1
                    labels[neighbor] = cluster_label
                    if core_samples[neighbor]
                        append!(stack, neighbors[neighbor])
                    end
                end
            end
        end
        
        # Store results
        dbscan.labels_ = labels
        dbscan.core_sample_indices_ = findall(core_samples)
        dbscan.components_ = X[dbscan.core_sample_indices_, :]
        dbscan.fitted = true
        
        return dbscan
    else
        # Predicting or transforming
        if y === nothing
            return dbscan.labels_
        elseif y == :transform
            return X[dbscan.core_sample_indices_, :]
        elseif y == :fit_predict
            return dbscan(X).labels_
        else
            throw(ArgumentError("Invalid second argument. Use nothing for prediction, :transform for transformation, or :fit_predict for fitting and predicting."))
        end
    end
end

"""
get_params(dbscan::DBSCAN)
Get parameters for this estimator.

# Returns

params::Dict: Parameter names mapped to their values.

# Examples
"""
# Helper functions
function get_params(dbscan::DBSCAN)
    return Dict(
        :eps => dbscan.eps,
        :min_samples => dbscan.min_samples,
        :metric => dbscan.metric,
        :metric_params => dbscan.metric_params,
        :algorithm => dbscan.algorithm,
        :leaf_size => dbscan.leaf_size,
        :p => dbscan.p,
        :n_jobs => dbscan.n_jobs
    )
end

"""
set_params!(dbscan::DBSCAN; kwargs...)
Set the parameters of this estimator.

# Arguments
kwargs...: Estimator parameters.

# Returns
dbscan::DBSCAN: The DBSCAN object.

# Examples
```julia
dbscan = DBSCAN()
set_params!(dbscan, eps=0.8, min_samples=15)
```
"""
function set_params!(dbscan::DBSCAN; kwargs...)
    for (key, value) in kwargs
        setproperty!(dbscan, key, value)
    end
    return dbscan
end

"""
Base.show(io::IO, dbscan::DBSCAN)
Custom show method for DBSCAN objects.

# Arguments
io::IO: The I/O stream to which the representation is written.
dbscan::DBSCAN: The DBSCAN object to be displayed.

# Examples
```julia
dbscan = DBSCAN(eps=0.7, min_samples=10)
println(dbscan)
```
"""
function Base.show(io::IO, dbscan::DBSCAN)
    params = get_params(dbscan)
    print(io, "DBSCAN(")
    for (i, (key, value)) in enumerate(params)
        if value !== nothing
            print(io, "$key=$value")
            if i < length(params)
                print(io, ", ")
            end
        end
    end
    print(io, ")")
end