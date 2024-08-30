function mean_squared_error(y_true::AbstractArray, y_pred::AbstractArray; 
                            sample_weight=nothing, multioutput="uniform_average")
    if size(y_true) != size(y_pred)
        throw(DimensionMismatch("The dimensions of y_true and y_pred must match"))
    end

    errors = (y_true .- y_pred).^2

    if sample_weight !== nothing
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
        else
            throw(ArgumentError("Invalid multioutput option. Choose 'uniform_average' or 'raw_values'"))
        end
    end
end

"""
    mse(y_true, y_pred; sample_weight=nothing, multioutput="uniform_average")

Alias for `mean_squared_error`.
"""
const mse = mean_squared_error