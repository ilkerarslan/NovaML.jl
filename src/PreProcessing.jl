module PreProcessing
    
using Statistics
import ..Nova: AbstractScaler


export StandardScaler


mutable struct StandardScaler <: AbstractScaler 
    mean::Vector{Float64}
    std::Vector{Float64}
    fitted::Bool

    StandardScaler() = new(Float64[], Float64[], false)
end


function (scaler::StandardScaler)(X::Matrix{<:Real})
    if scaler.fitted == false
        scaler.mean = vec(mean(X, dims=1))
        scaler.std = vec(std(X, dims=1))
        scaler.fitted = true
    else
        return (X .- scaler.mean') ./ scaler.std'
    end
end


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
    else
        return (X .- scaler.min') ./ (scaler.max .- scaler.min)'
    end
end

end