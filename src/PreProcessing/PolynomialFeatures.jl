using LinearAlgebra
using SparseArrays

mutable struct PolynomialFeatures
    degree::Union{Int, Tuple{Int, Int}}
    interaction_only::Bool
    include_bias::Bool
    order::Symbol
    n_features_in_::Int
    n_output_features_::Int
    powers_::Matrix{Int}
    feature_names_in_::Vector{String}
    fitted::Bool

    function PolynomialFeatures(;
        degree::Union{Int, Tuple{Int, Int}} = 2,
        interaction_only::Bool = false,
        include_bias::Bool = true,
        order::Symbol = :C
    )
        new(degree, interaction_only, include_bias, order, 0, 0, Matrix{Int}(undef, 0, 0), String[], false)
    end
end

function (poly::PolynomialFeatures)(X::AbstractMatrix, y=nothing)
    if !poly.fitted
        return fit_transform(poly, X)
    else
        return transform(poly, X)
    end
end

function fit_transform(poly::PolynomialFeatures, X::AbstractMatrix)
    n_samples, n_features = size(X)
    poly.n_features_in_ = n_features
    
    if typeof(poly.degree) <: Tuple
        min_degree, max_degree = poly.degree
    else
        min_degree, max_degree = 0, poly.degree
    end

    combinations = generate_combinations(n_features, max_degree, poly.interaction_only, poly.include_bias)
    poly.powers_ = reduce(vcat, combinations') 
    poly.n_output_features_ = size(poly.powers_, 1)

    poly.feature_names_in_ = ["x$i" for i in 1:n_features]
    poly.fitted = true

    return transform(poly, X)
end

function transform(poly::PolynomialFeatures, X::AbstractMatrix)
    n_samples, n_features = size(X)
    
    if n_features != poly.n_features_in_
        throw(DimensionMismatch("X has $n_features features, but PolynomialFeatures is expecting $(poly.n_features_in_) features."))
    end

    if issparse(X)
        return _transform_sparse(poly, X)
    else
        return _transform_dense(poly, X)
    end
end

function _transform_dense(poly::PolynomialFeatures, X::AbstractMatrix)
    n_samples, _ = size(X)
    XP = ones(n_samples, poly.n_output_features_)
    
    for (i, powers) in enumerate(eachrow(poly.powers_))
        if all(powers .== 0)
            continue  # Skip the bias term
        end
        XP[:, i] = prod(X[:, j].^p for (j, p) in enumerate(powers) if p != 0)
    end

    return poly.order == :F ? permutedims(XP) : XP
end

function _transform_sparse(poly::PolynomialFeatures, X::SparseMatrixCSC)
    n_samples, _ = size(X)
    
    if maximum(poly.powers_) > 3
        X = SparseMatrixCSC(X)  # Convert to CSC if degree > 3
    else
        X = SparseMatrixCSR(X)  # Use CSR for faster processing otherwise
    end

    rows, cols, data = Int[], Int[], Float64[]
    
    for (sample_idx, sample) in enumerate(eachrow(X))
        sample_data = Dict(zip(sample.nzind, sample.nzval))
        for (feat_idx, powers) in enumerate(eachrow(poly.powers_))
            value = 1.0
            for (col, power) in enumerate(powers)
                if power != 0
                    value *= get(sample_data, col, 0.0)^power
                end
            end
            if value != 0
                push!(rows, sample_idx)
                push!(cols, feat_idx)
                push!(data, value)
            end
        end
    end

    XP = sparse(rows, cols, data, n_samples, poly.n_output_features_)
    return poly.order == :F ? SparseMatrixCSC(XP') : XP
end

function generate_combinations(n_features, degree, interaction_only, include_bias)
    combinations = Vector{Int}[]
    for d in (include_bias ? 0 : 1):degree
        for c in combinations_with_replacement(1:n_features, d)
            if !interaction_only || length(unique(c)) == length(c)
                powers = zeros(Int, n_features)
                for i in c
                    powers[i] += 1
                end
                push!(combinations, powers)
            end
        end
    end
    return combinations
end

function combinations_with_replacement(iterable, r)
    pool = collect(iterable)
    n = length(pool)
    if r == 0
        return [Int[]]
    end
    indices = ones(Int, r)
    return Channel() do ch
        while true
            put!(ch, pool[indices])
            for i in r:-1:1
                if indices[i] != n
                    indices[i] += 1
                    for j in i+1:r
                        indices[j] = indices[i]
                    end
                    break
                end
                i == 1 && return
            end
        end
    end
end

function get_feature_names_out(poly::PolynomialFeatures, input_features=nothing)
    if !poly.fitted
        throw(ErrorException("PolynomialFeatures is not fitted. Call the transformer with data first."))
    end

    if input_features === nothing
        input_features = poly.feature_names_in_
    elseif length(input_features) != poly.n_features_in_
        throw(DimensionMismatch("input_features has $(length(input_features)) features, but PolynomialFeatures is expecting $(poly.n_features_in_) features."))
    end

    feature_names = String[]
    for powers in eachrow(poly.powers_)
        name = join(["$(input_features[i])^$p" for (i, p) in enumerate(powers) if p != 0], " ")
        name = isempty(name) ? "1" : name
        push!(feature_names, name)
    end

    return feature_names
end

# Helper method to get the degree of the polynomial for a specific feature combination
function get_degree(powers::AbstractVector{Int})
    return sum(powers)
end

function Base.show(io::IO, poly::PolynomialFeatures)
    print(io, "PolynomialFeatures(degree=$(poly.degree), ",
        "interaction_only=$(poly.interaction_only), ",
        "include_bias=$(poly.include_bias), ",
        "order=:$(poly.order), ",
        "fitted=$(poly.fitted))")
end