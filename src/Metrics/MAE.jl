function mean_absolute_error(y_true::AbstractArray, y_pred::AbstractArray; 
                             sample_weight=nothing, multioutput="uniform_average")
    if size(y_true) != size(y_pred)
        throw(DimensionMismatch("The dimensions of y_true and y_pred must match"))
    end

    errors = abs.(y_true .- y_pred)

    if sample_weight !== nothing
        if length(sample_weight) != size(y_true, 1)
            throw(DimensionMismatch("The length of sample_weight must match the number of samples"))
        end
        errors .*= sample_weight
    end

    if ndims(y_true) == 1 || (ndims(y_true) == 2 && size(y_true, 2) == 1)
        return mean(errors)
    else
        axis_errors = mean(errors, dims=1)
        if multioutput == "uniform_average"
            return mean(axis_errors)
        elseif multioutput == "raw_values"
            return vec(axis_errors)
        elseif multioutput isa AbstractArray
            if length(multioutput) != size(y_true, 2)
                throw(DimensionMismatch("The length of multioutput weights must match the number of outputs"))
            end
            return sum(axis_errors .* multioutput)
        else
            throw(ArgumentError("Invalid multioutput option. Choose 'uniform_average', 'raw_values', or provide an array of weights"))
        end
    end
end

const mae = mean_absolute_error