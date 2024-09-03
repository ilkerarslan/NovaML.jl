mutable struct OneHotEncoder
    categories::Vector{Vector}
    category_maps::Vector{Dict}
    n_features::Int
    fitted::Bool

    OneHotEncoder() = new([], [], 0, false)
end

function (encoder::OneHotEncoder)(X::Union{AbstractVector, AbstractMatrix})
    if X isa AbstractVector
        X = reshape(X, :, 1)
    end

    if !encoder.fitted        
        encoder.n_features = size(X, 2)
        encoder.categories = [unique(sort(X[:, i])) for i in 1:encoder.n_features]
        encoder.category_maps = [Dict(val => i for (i, val) in enumerate(cats)) for cats in encoder.categories]
        encoder.fitted = true
    end
    
    return _transform(encoder, X)
end

function (encoder::OneHotEncoder)(X::Union{AbstractVector, AbstractMatrix}, type::Symbol)
    if type == :inverse
        if X isa AbstractVector
            X = reshape(X, :, 1)
        end    
        return _inverse_transform(encoder, X)
    else
        throw(ErrorException("Type can be :inverse"))
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

function Base.show(io::IO, encoder::OneHotEncoder)
    fitted_status = encoder.fitted ? "fitted" : "not fitted"
    n_features = encoder.n_features
    print(io, "OneHotEncoder($(fitted_status), $(n_features) features)")
end