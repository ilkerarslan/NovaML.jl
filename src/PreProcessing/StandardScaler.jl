using Statistics
import ...NovaML: AbstractScaler

mutable struct StandardScaler <: AbstractScaler 
    mean::Vector{Float64}
    std::Vector{Float64}
    fitted::Bool

    StandardScaler() = new(Float64[], Float64[], false)
end

function (scaler::StandardScaler)(X::AbstractVecOrMat{<:Real}; type::Symbol=:transform)
    if !scaler.fitted
        if X isa AbstractVector
            scaler.mean = [mean(X)]
            scaler.std = [std(X)]
        else
            scaler.mean = vec(mean(X, dims=1))
            scaler.std = vec(std(X, dims=1))
        end
        scaler.fitted = true
    end

    if type == :transform
        if X isa AbstractVector
            return (X .- scaler.mean[1]) ./ scaler.std[1]
        else
            return (X .- scaler.mean') ./ scaler.std'
        end
    elseif type == :inverse_transform
        if !scaler.fitted
            throw(ErrorException("StandardScaler is not fitted. Call the scaler with data to fit before using inverse_transform."))
        end
        if X isa AbstractVector
            return X .* scaler.std[1] .+ scaler.mean[1]
        else
            return X .* scaler.std' .+ scaler.mean'
        end
    else
        throw(ArgumentError("Invalid type. Use :transform or :inverse_transform."))
    end
end