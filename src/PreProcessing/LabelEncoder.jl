mutable struct LabelEncoder
    classes::Vector
    class_dict::Dict
    fitted::Bool

    LabelEncoder() = new([], Dict(), false)
end

function (encoder::LabelEncoder)(y::AbstractVector)
    if !encoder.fitted
        encoder.classes = sort(unique(y))
        encoder.class_dict = Dict(class => i-1 for (i, class) in enumerate(encoder.classes))
        encoder.fitted = true
    end
    return [get(encoder.class_dict, label, -1) for label in y]
end

function (encoder::LabelEncoder)(y::AbstractVector, type::Symbol)
    if type == :inverse_transform
        if !encoder.fitted
            throw(ErrorException("LabelEncoder is not fitted. Call encoder(y) to fit and transform the data first."))
        end
        reverse_dict = Dict(v => k for (k, v) in encoder.class_dict)
        return [get(reverse_dict, label, nothing) for label in y]
    else 
        throw(ErrorException("Type can only be :inverse_transform"))
    end
end

function Base.show(io::IO, encoder::LabelEncoder)
    fitted_status = encoder.fitted ? "fitted" : "not fitted"
    n_classes = length(encoder.classes)
    print(io, "LabelEncoder($(fitted_status), $(n_classes) classes)")
end