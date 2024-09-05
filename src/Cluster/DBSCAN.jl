using LinearAlgebra
using SparseArrays
using ...NovaML.Neighbors: KNeighborsClassifier
using Distances

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

function (dbscan::DBSCAN)(X::AbstractMatrix, y=nothing; sample_weight=nothing)
    if !dbscan.fitted
        n_samples, n_features = size(X)
        dbscan.n_features_in_ = n_features
        
        if sample_weight === nothing
            sample_weight = ones(n_samples)
        end
        
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
        
        for i in 1:n_samples
            if sum(sample_weight[neighbors[i]]) >= dbscan.min_samples
                core_samples[i] = true
            end
        end
        
        cluster_label = 0
        for i in 1:n_samples
            if labels[i] != -1 || !core_samples[i]
                continue
            end
            
            cluster_label += 1
            labels[i] = cluster_label
            
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
        
        dbscan.labels_ = labels
        dbscan.core_sample_indices_ = findall(core_samples)
        dbscan.components_ = X[dbscan.core_sample_indices_, :]
        dbscan.fitted = true
        
        return dbscan
    else
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

function set_params!(dbscan::DBSCAN; kwargs...)
    for (key, value) in kwargs
        setproperty!(dbscan, key, value)
    end
    return dbscan
end

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