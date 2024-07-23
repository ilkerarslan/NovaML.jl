function accuracy_score(y::AbstractVector, ŷ::AbstractVector)
    if length(ŷ) != length(y)
        throw("The length of y and ŷ must be the same.")
    end

    correct = sum(y .== ŷ)
    total = length(y)

    return correct / total
end