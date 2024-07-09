"""
    OneHotEncoder

Encode categorical features as a one-hot numeric array.

# Fields
- `categories::Vector{Vector}`: List of arrays, each containing the unique categories for a feature
- `category_maps::Vector{Dict}`: List of dictionaries mapping categories to their encoded indices
- `n_features::Int`: Number of features
- `fitted::Bool`: Indicates whether the encoder has been fitted
"""
mutable struct OneHotEncoder
    categories::Vector{Vector}
    category_maps::Vector{Dict}
    n_features::Int
    fitted::Bool

    OneHotEncoder() = new([], [], 0, false)
end

"""
    (encoder::OneHotEncoder)(X::Union{AbstractVector, AbstractMatrix})

Fit the OneHotEncoder to the input features.

# Arguments
- `X::Union{AbstractVector, AbstractMatrix}`: Vector or matrix of features to encode

# Returns
- The fitted OneHotEncoder object
"""
function (encoder::OneHotEncoder)(X::Union{AbstractVector, AbstractMatrix})
    if X isa AbstractVector
        X = reshape(X, :, 1)
    end
    
    encoder.n_features = size(X, 2)
    encoder.categories = [unique(sort(X[:, i])) for i in 1:encoder.n_features]
    encoder.category_maps = [Dict(val => i for (i, val) in enumerate(cats)) for cats in encoder.categories]
    encoder.fitted = true
    return encoder
end

"""
    (encoder::OneHotEncoder)(X::Union{AbstractVector, AbstractMatrix}, mode::Symbol)

Transform features or inverse transform encoded features.

# Arguments
- `X::Union{AbstractVector, AbstractMatrix}`: Vector or matrix to transform or inverse transform
- `mode::Symbol`: Either :transform or :inverse_transform

# Returns
- Transformed or inverse transformed features

# Throws
- `ErrorException`: If the encoder hasn't been fitted or if an invalid mode is provided
"""
function (encoder::OneHotEncoder)(X::Union{AbstractVector, AbstractMatrix}, mode::Symbol)
    if !encoder.fitted
        throw(ErrorException("OneHotEncoder must be fitted before transforming or inverse transforming."))
    end

    if X isa AbstractVector
        X = reshape(X, :, 1)
    end

    if mode == :transform
        return _transform(encoder, X)
    elseif mode == :inverse_transform
        return _inverse_transform(encoder, X)
    else
        throw(ErrorException("Invalid mode. Use :transform or :inverse_transform."))
    end
end

function _transform(encoder::OneHotEncoder, X::AbstractMatrix)
    n_samples = size(X, 1)
    result = zeros(Float64, n_samples, sum(length(cats) for cats in encoder.categories))
    col_offset = 1
    for (feature, (categories, category_map)) in enumerate(zip(encoder.categories, encoder.category_maps))
        for (row, val) in enumerate(X[:, feature])
            col = get(category_map, val, 0)
            if col != 0
                result[row, col_offset + col - 1] = 1.0
            end
        end
        col_offset += length(categories)
    end
    return result
end

function _inverse_transform(encoder::OneHotEncoder, X::AbstractMatrix)
    n_samples = size(X, 1)
    result = Matrix{Any}(undef, n_samples, encoder.n_features)
    col_offset = 1
    for (feature, categories) in enumerate(encoder.categories)
        n_categories = length(categories)
        feature_slice = X[:, col_offset:col_offset+n_categories-1]
        for row in 1:n_samples
            cat_index = argmax(feature_slice[row, :])
            result[row, feature] = categories[cat_index]
        end
        col_offset += n_categories
    end
    return encoder.n_features == 1 ? vec(result) : result
end

"""
    Base.show(io::IO, encoder::OneHotEncoder)

Custom pretty printing for OneHotEncoder.

# Arguments
- `io::IO`: The I/O stream
- `encoder::OneHotEncoder`: The OneHotEncoder object to be printed
"""
function Base.show(io::IO, encoder::OneHotEncoder)
    fitted_status = encoder.fitted ? "fitted" : "not fitted"
    n_features = encoder.n_features
    print(io, "OneHotEncoder($(fitted_status), $(n_features) features)")
end