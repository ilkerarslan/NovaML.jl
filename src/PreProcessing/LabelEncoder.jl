"""
    LabelEncoder

Encode categorical labels with value between 0 and n_classes-1.

# Fields
- `classes::Vector`: Holds the unique classes found during fitting
- `class_dict::Dict`: Dictionary mapping classes to their encoded values
- `fitted::Bool`: Indicates whether the encoder has been fitted
"""
mutable struct LabelEncoder
    classes::Vector
    class_dict::Dict
    fitted::Bool

    LabelEncoder() = new([], Dict(), false)
end

"""
    (encoder::LabelEncoder)(y::AbstractVector)

Fit the LabelEncoder to the input labels.

# Arguments
- `y::AbstractVector`: Vector of labels to encode

# Returns
- The fitted LabelEncoder object
"""
function (encoder::LabelEncoder)(y::AbstractVector)
    encoder.classes = sort(unique(y))
    encoder.class_dict = Dict(class => i-1 for (i, class) in enumerate(encoder.classes))
    encoder.fitted = true
    return encoder
end

"""
    (encoder::LabelEncoder)(y::AbstractVector, mode::Symbol)

Transform labels or inverse transform encoded labels.

# Arguments
- `y::AbstractVector`: Vector of labels to transform or inverse transform
- `mode::Symbol`: Either :transform or :inverse_transform

# Returns
- Transformed or inverse transformed labels

# Throws
- `ErrorException`: If the encoder hasn't been fitted or if an invalid mode is provided
"""
function (encoder::LabelEncoder)(y::AbstractVector, mode::Symbol)
    if !encoder.fitted
        throw(ErrorException("LabelEncoder must be fitted before transforming or inverse transforming."))
    end

    if mode == :transform
        return [get(encoder.class_dict, label, -1) for label in y]
    elseif mode == :inverse_transform
        reverse_dict = Dict(v => k for (k, v) in encoder.class_dict)
        return [get(reverse_dict, Int(label), nothing) for label in y]
    else
        throw(ErrorException("Invalid mode. Use :transform or :inverse_transform."))
    end
end

"""
    Base.show(io::IO, encoder::LabelEncoder)

Custom pretty printing for LabelEncoder.

# Arguments
- `io::IO`: The I/O stream
- `encoder::LabelEncoder`: The LabelEncoder object to be printed
"""
function Base.show(io::IO, encoder::LabelEncoder)
    fitted_status = encoder.fitted ? "fitted" : "not fitted"
    n_classes = length(encoder.classes)
    print(io, "LabelEncoder($(fitted_status), $(n_classes) classes)")
end