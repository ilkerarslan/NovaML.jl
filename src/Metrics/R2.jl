using Statistics

function r2_score(y_true::AbstractArray, y_pred::AbstractArray; 
                  sample_weight=nothing, multioutput="uniform_average")
    if size(y_true) != size(y_pred)
        throw(DimensionMismatch("The dimensions of y_true and y_pred must match"))
    end

    if ndims(y_true) == 1
        y_true = reshape(y_true, :, 1)
        y_pred = reshape(y_pred, :, 1)
    end

    n_samples, n_outputs = size(y_true)
    weights = sample_weight !== nothing ? sample_weight : ones(n_samples)
    
    if length(weights) != n_samples
        throw(DimensionMismatch("The length of sample_weight must match the number of samples"))
    end
    
    # Calculate weighted means
    weighted_mean_true = sum(weights .* y_true, dims=1) ./ sum(weights)
    
    numerator = sum(weights .* (y_true .- y_pred).^2, dims=1)
    denominator = sum(weights .* (y_true .- weighted_mean_true).^2, dims=1)
    
    r2 = 1 .- numerator ./ denominator
    r2[denominator .== 0] .= 0

    if n_outputs == 1
        return r2[1]
    elseif multioutput == "uniform_average"
        return mean(r2)
    elseif multioutput == "raw_values"
        return vec(r2)
    else
        throw(ArgumentError("Invalid multioutput option. Choose 'uniform_average' or 'raw_values'"))
    end
end

function adj_r2_score(y_true::AbstractArray, y_pred::AbstractArray; n_features::Union{Int,Nothing}=nothing, 
                      sample_weight=nothing, multioutput="uniform_average", n_features_kw::Union{Int,Nothing}=nothing)
    if size(y_true) != size(y_pred)
        throw(DimensionMismatch("The dimensions of y_true and y_pred must match"))
    end

    if ndims(y_true) == 1
        y_true = reshape(y_true, :, 1)
        y_pred = reshape(y_pred, :, 1)
    end

    n_samples, n_outputs = size(y_true)

    # Use n_features_kw if provided, otherwise use n_features
    n_features_final = n_features_kw !== nothing ? n_features_kw : n_features

    if n_features_final === nothing
        throw(ArgumentError("Number of features (n_features) must be provided"))
    end

    if n_samples <= n_features_final + 1
        throw(ArgumentError("The number of samples must be greater than the number of features plus one"))
    end

    r2 = r2_score(y_true, y_pred; sample_weight=sample_weight, multioutput="raw_values")
    
    adj_r2 = 1 .- (1 .- r2) .* (n_samples - 1) ./ (n_samples - n_features_final - 1)

    if n_outputs == 1
        return adj_r2[1]
    elseif multioutput == "uniform_average"
        return mean(adj_r2)
    elseif multioutput == "raw_values"
        return vec(adj_r2)
    else
        throw(ArgumentError("Invalid multioutput option. Choose 'uniform_average' or 'raw_values'"))
    end
end