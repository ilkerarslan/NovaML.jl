using Statistics
import ...NovaML: AbstractScaler

mutable struct MinMaxScaler <: AbstractScaler
    min::Vector{Float64}
    max::Vector{Float64}
    fitted::Bool

    MinMaxScaler() = new(Float64[], Float64[], false)
end

function (scaler::MinMaxScaler)(X::Matrix{<:Real})
    if scaler.fitted == false
        scaler.min = vec(minimum(X, dims=1))
        scaler.max = vec(maximum(X, dims=1))
        scaler.fitted = true
    end
    return (X .- scaler.min') ./ (scaler.max .- scaler.min)'
end