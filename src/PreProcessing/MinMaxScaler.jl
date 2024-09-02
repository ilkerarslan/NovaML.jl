using Statistics
import ...NovaML: AbstractScaler

mutable struct MinMaxScaler <: AbstractScaler
    min::Vector{Float64}
    max::Vector{Float64}
    fitted::Bool

    MinMaxScaler() = new(Float64[], Float64[], false)
end

function (scaler::MinMaxScaler)(X::AbstractVecOrMat{<:Real}; type::Symbol=:transform)
    if !scaler.fitted
        if X isa AbstractVector
            scaler.min = [minimum(X)]
            scaler.max = [maximum(X)]
        else
            scaler.min = vec(minimum(X, dims=1))
            scaler.max = vec(maximum(X, dims=1))
        end
        scaler.fitted = true
    end

    if type == :transform
        if X isa AbstractVector
            return (X .- scaler.min[1]) ./ (scaler.max[1] - scaler.min[1])
        else
            return (X .- scaler.min') ./ (scaler.max .- scaler.min)'
        end
    elseif type == :inverse
        if !scaler.fitted
            throw(ErrorException("MinMaxScaler is not fitted. Call the scaler with data to fit before using inverse transform."))
        end
        if X isa AbstractVector
            return X .* (scaler.max[1] - scaler.min[1]) .+ scaler.min[1]
        else
            return X .* (scaler.max .- scaler.min)' .+ scaler.min'
        end
    else
        throw(ArgumentError("Invalid type. Use :transform or :inverse."))
    end
end