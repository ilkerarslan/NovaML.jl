# src/PreProcessing/ColumnTransformer.jl

export ColumnTransformer

mutable struct ColumnTransformer
    transformers::Vector{Tuple{String, Any, Union{Vector{Int}, Vector{String}}}}
    sparse::Bool
    remainder::Symbol  # :drop, :passthrough
    sparse_threshold::Float64
    n_features_in_::Union{Int, Nothing}
    feature_names_in_::Union{Vector{String}, Nothing}
    _feature_indices::Union{Dict{String, Vector{Int}}, Nothing}
    _column_dtype_masks::Union{Dict{String, BitVector}, Nothing}
    fitted::Bool

    function ColumnTransformer(
        transformers::Vector{<:Tuple};
        sparse::Bool=false,
        remainder::Symbol=:drop,
        sparse_threshold::Float64=0.3
    )
        @assert remainder in [:drop, :passthrough] "remainder must be either :drop or :passthrough"
        @assert 0.0 <= sparse_threshold <= 1.0 "sparse_threshold must be between 0 and 1"
        
        new(
            transformers,
            sparse,
            remainder,
            sparse_threshold,
            nothing,
            nothing,
            nothing,
            nothing,
            false
        )
    end
end

function _validate_transformers(ct::ColumnTransformer, X::AbstractMatrix)
    n_features = size(X, 2)
    all_columns = Set(1:n_features)
    selected_columns = Set{Int}()
    
    for (name, transformer, columns) in ct.transformers
        # Convert column names to indices if necessary
        if eltype(columns) <: AbstractString
            if ct.feature_names_in_ === nothing
                throw(ArgumentError("Feature names not available for string-based column selection"))
            end
            column_indices = indexin(columns, ct.feature_names_in_)
            if any(isnothing, column_indices)
                throw(ArgumentError("Some column names not found in feature names"))
            end
            columns = column_indices
        end
        
        # Check for overlapping columns
        overlap = intersect(Set(columns), selected_columns)
        if !isempty(overlap)
            throw(ArgumentError("Columns must not overlap, found duplicate columns: $overlap"))
        end
        
        union!(selected_columns, columns)
    end
    
    # Validate remainder columns
    if ct.remainder == :passthrough
        remainder_columns = setdiff(all_columns, selected_columns)
        if !isempty(remainder_columns)
            push!(ct.transformers, ("remainder", PassthroughTransformer(), collect(remainder_columns)))
        end
    end
end

function (ct::ColumnTransformer)(X::AbstractMatrix, y=nothing)
    n_samples, n_features = size(X)
    
    if !ct.fitted
        ct.n_features_in_ = n_features
        ct.feature_names_in_ = ["feature_$i" for i in 1:n_features]
        _validate_transformers(ct, X)
        ct.fitted = true
    end
    
    # Transform each subset of columns
    transformed_blocks = []
    
    for (name, transformer, columns) in ct.transformers
        X_subset = X[:, columns]
        if y === nothing
            # Transform
            transformed = transformer(X_subset)
        else
            # Fit and transform
            transformed = transformer(X_subset, y)
        end
        push!(transformed_blocks, transformed)
    end
    
    # Horizontally concatenate all transformed blocks
    result = hcat(transformed_blocks...)
    
    # Convert to sparse if needed
    if ct.sparse && !(result isa SparseMatrixCSC)
        n_zeros = count(iszero, result)
        sparsity = n_zeros / (size(result, 1) * size(result, 2))
        if sparsity >= ct.sparse_threshold
            result = sparse(result)
        end
    end
    
    return result
end

# Utility transformer for passthrough columns
struct PassthroughTransformer end
(t::PassthroughTransformer)(X::AbstractMatrix, y=nothing) = X

# Helper function to get transformed feature names
function get_feature_names_out(ct::ColumnTransformer)
    if !ct.fitted
        throw(ErrorException("ColumnTransformer is not fitted. Call with data first."))
    end
    
    feature_names = String[]
    for (name, transformer, columns) in ct.transformers
        if hasmethod(get_feature_names_out, (typeof(transformer),))
            # If transformer supports get_feature_names_out, use it
            subset_names = get_feature_names_out(transformer)
            append!(feature_names, subset_names)
        else
            # Otherwise, generate generic names
            n_features_out = size(transformer(X[:, columns]), 2)
            append!(feature_names, ["$(name)_$(i)" for i in 1:n_features_out])
        end
    end
    
    return feature_names
end