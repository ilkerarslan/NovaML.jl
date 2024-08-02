# Activation functions
"""Linear activation function"""
linearactivation(X) = X
"""Sigmoid activation function for scalar value"""
sigmoid(z::Real) = 1 / (1 + exp(-1*clamp(z, -250, 250)))
"""Sigmoid activation function for an array"""
sigmoid(z::AbstractArray) = 1 ./ (1 .+ exp.(-1*clamp.(z, -250, 250)))

"""Compute logistic sigmoid activation"""
sigmoid(z) = 1. / (1. + exp(-1*clamp(z, -250, 250)))

logit(p) = log(p / (1 - p))

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

# Default scoring function
function default_score(y, y_pred)
    if y isa AbstractVector{<:Number} && y_pred isa AbstractVector{<:Number}
        # R-squared for regression
        ss_res = sum((y .- y_pred).^2)
        ss_tot = sum((y .- mean(y)).^2)
        return max(0, 1 - ss_res / ss_tot)  # Ensure non-negative R-squared
    else
        # Accuracy for classification
        return sum(y .== y_pred) / length(y)
    end
end

# Common functions in GridSearchCV and RandomSearchCV 
function _generate_param_combinations(param_grid)
    all_params = []
    for grid in param_grid
        component = grid[1]
        param_list = grid[2:end]
        params = _param_product(component, param_list)
        append!(all_params, params)
    end
    return all_params
end

function _param_product(component, param_list)
    keys = [p[1] for p in param_list]
    values = [p[2] for p in param_list]
    combinations = Iterators.product(values...)
    return [[component, zip(keys, combo)...] for combo in combinations]
end

function _set_params!(params)
    component = params[1]
    for (param, value) in params[2:end]
        setproperty!(component, param, value)
    end
end