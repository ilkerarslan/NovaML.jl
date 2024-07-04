module PreProcessing
    
using Statistics

export StandardScaler

abstract type AbstractScaler end

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

end