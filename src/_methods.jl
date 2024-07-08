# Activation functions
"""Linear activation function"""
linearactivation(X) = X
"""Sigmoid activation function for scalar value"""
sigmoid(z::Real) = 1 / (1 + exp(-1*clamp(z, -250, 250)))
"""Sigmoid activation function for an array"""
sigmoid(z::AbstractArray) = 1 ./ (1 .+ exp.(-1*clamp.(z, -250, 250)))

"""Compute logistic sigmoid activation"""
sigmoid(z) = 1. / (1. + exp(-1*clamp(z, -250, 250)))

# Common methods
"""Calculate net input for AbstractModel"""
net_input(m::AbstractModel, x::AbstractVector) = x'*m.w + m.b
net_input(m::AbstractModel, X::AbstractMatrix) = X * m.w .+ m.b
# net_input for MulticlassPerceptron
"""Calculate net input for AbstractMultiClass"""
net_input(m::AbstractMultiClass, x::AbstractVector) = m.W' * x .+ m.b
net_input(m::AbstractMultiClass, X::AbstractMatrix) = X * m.W .+ m.b'

#Softmax function for probability distribution
function softmax(X::AbstractMatrix)
    exp_X = exp.(X .- maximum(X, dims=2))
    return exp_X ./ sum(exp_X, dims=2)
end
