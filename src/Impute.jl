module Impute

using Statistics
using StatsBase

abstract type AbstractImputer end

export SimpleImputer

mutable struct SimpleImputer <: AbstractImputer
    strategy::Symbol
    fill_value::Union{Number, String, Nothing}
    missing_values::Any
    imputation_values::Union{Dict{Int, Union{Number, String}}, Nothing}
    fitted::Bool

    function SimpleImputer(;
        strategy::Symbol=:mean,
        fill_value::Union{Number, String, Nothing}=nothing,
        missing_values::Any=missing
    )
        if !(strategy in [:mean, :median, :most_frequent, :constant])
            throw(ArgumentError("strategy must be one of :mean, :median, :most_frequent, or :constant"))
        end
        
        if strategy == :constant && fill_value === nothing
            throw(ArgumentError("fill_value must be specified when strategy is :constant"))
        end

        new(strategy, fill_value, missing_values, nothing, false)
    end
end

function (imputer::SimpleImputer)(X::AbstractMatrix)
    if !imputer.fitted
        n_features = size(X, 2)
        imputer.imputation_values = Dict{Int, Union{Number, String}}()
    
        for col in 1:n_features
            column = X[:, col]
            non_missing = filter(!ismissing, column)
    
            if imputer.strategy == :mean
                imputer.imputation_values[col] = mean(non_missing)
            elseif imputer.strategy == :median
                imputer.imputation_values[col] = median(non_missing)
            elseif imputer.strategy == :most_frequent
                imputer.imputation_values[col] = mode(non_missing)
            elseif imputer.strategy == :constant
                imputer.imputation_values[col] = imputer.fill_value
            end
        end
    
        imputer.fitted = true
    end

    X_imputed = copy(X)

    for (col, value) in imputer.imputation_values
        mask = ismissing.(X[:, col])
        X_imputed[mask, col] .= value
    end

    return X_imputed
end


function Base.show(io::IO, imputer::SimpleImputer)
    println(io, "SimpleImputer(")
    println(io, "  strategy=$(imputer.strategy),")
    println(io, "  fill_value=$(imputer.fill_value),")
    println(io, "  missing_values=$(imputer.missing_values),")
    println(io, "  fitted=$(imputer.fitted)")
    print(io, ")")
end

end # module Impute