abstract type AbstractModel end
abstract type AbstractMultiClass <: AbstractModel end

"""AbstractModel batch prediction"""
(m::AbstractModel)(X::AbstractMatrix) = [m(x) for x in eachrow(X)]

abstract type AbstractScaler end