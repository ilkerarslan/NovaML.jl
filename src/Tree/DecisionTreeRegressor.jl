using Random
using Statistics

import ...NovaML: AbstractModel

export DecisionTreeRegressor

mutable struct RegressorNode
    feature_index::Union{Int, Nothing}
    threshold::Union{Float64, Nothing}
    left::Union{RegressorNode, Nothing}
    right::Union{RegressorNode, Nothing}
    is_leaf::Bool
    value::Union{Float64, Nothing}

    RegressorNode() = new(nothing, nothing, nothing, nothing, false, nothing)
end

mutable struct DecisionTreeRegressor <: AbstractModel
    max_depth::Union{Int, Nothing}
    min_samples_split::Int
    min_samples_leaf::Int
    min_weight_fraction_leaf::Float64
    max_features::Union{Int, Float64, String, Nothing}
    random_state::Union{Int, Nothing}
    max_leaf_nodes::Union{Int, Nothing}
    min_impurity_decrease::Float64
    ccp_alpha::Float64
    criterion::String
    splitter::String
    root::Union{RegressorNode, Nothing}
    n_features_::Int
    feature_importances_::Union{Vector{Float64}, Nothing}
    fitted::Bool

    function DecisionTreeRegressor(;
        criterion::String="squared_error",
        splitter::String="best",
        max_depth::Union{Int, Nothing}=nothing,
        min_samples_split::Int=2,
        min_samples_leaf::Int=1,
        min_weight_fraction_leaf::Float64=0.0,
        max_features::Union{Int, Float64, String, Nothing}=nothing,
        random_state::Union{Int, Nothing}=nothing,
        max_leaf_nodes::Union{Int, Nothing}=nothing,
        min_impurity_decrease::Float64=0.0,
        ccp_alpha::Float64=0.0
    )
        new(
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_weight_fraction_leaf,
            max_features,
            random_state,
            max_leaf_nodes,
            min_impurity_decrease,
            ccp_alpha,
            criterion,
            splitter,
            nothing,
            0,
            nothing,
            false
        )
    end
end

function (tree::DecisionTreeRegressor)(X::AbstractMatrix, y::AbstractVector)
    n_samples, n_features = size(X)
    tree.n_features_ = n_features

    if tree.random_state !== nothing
        Random.seed!(tree.random_state)
    end

    tree.root = grow_tree(tree, X, y, 0)
    tree.feature_importances_ = compute_feature_importances(tree)
    tree.fitted = true
    return tree
end

function (tree::DecisionTreeRegressor)(X::AbstractMatrix)
    if !tree.fitted
        throw(ErrorException("This DecisionTreeRegressor instance is not fitted yet. Call the model with training data before using it for predictions."))
    end

    return [predict_sample(tree, tree.root, x) for x in eachrow(X)]
end

function grow_tree(tree::DecisionTreeRegressor, X::AbstractMatrix, y::AbstractVector, depth::Int)
    n_samples, n_features = size(X)
    
    # Check stopping criteria
    if (tree.max_depth !== nothing && depth >= tree.max_depth) ||
       n_samples < tree.min_samples_split ||
       n_samples < 2 * tree.min_samples_leaf ||
       (tree.max_leaf_nodes !== nothing && tree.max_leaf_nodes <= count_leaves(tree.root))
        return create_leaf(y)
    end
    
    # Find the best split
    best_feature, best_threshold = find_best_split(tree, X, y)
    
    if best_feature === nothing
        return create_leaf(y)
    end
    
    # Split the data
    left_mask = X[:, best_feature] .<= best_threshold
    right_mask = .!left_mask
    
    # Create the node
    node = RegressorNode()
    node.feature_index = best_feature
    node.threshold = best_threshold
    
    # Recursively build the left and right subtrees
    node.left = grow_tree(tree, X[left_mask, :], y[left_mask], depth + 1)
    node.right = grow_tree(tree, X[right_mask, :], y[right_mask], depth + 1)
    
    return node
end

function find_best_split(tree::DecisionTreeRegressor, X::AbstractMatrix, y::AbstractVector)
    n_samples, n_features = size(X)
    best_impurity_decrease = -Inf
    best_feature = nothing
    best_threshold = nothing
    
    features = 1:n_features
    if tree.max_features !== nothing
        if typeof(tree.max_features) == Int
            n_features_to_consider = min(tree.max_features, n_features)
        elseif typeof(tree.max_features) == Float64
            n_features_to_consider = round(Int, tree.max_features * n_features)
        elseif tree.max_features == "sqrt"
            n_features_to_consider = round(Int, sqrt(n_features))
        elseif tree.max_features == "log2"
            n_features_to_consider = round(Int, log2(n_features))
        end
        features = Random.shuffle(features)[1:n_features_to_consider]
    end
    
    for feature in features
        thresholds = unique(X[:, feature])
        
        for threshold in thresholds
            left_mask = X[:, feature] .<= threshold
            right_mask = .!left_mask
            
            if sum(left_mask) < tree.min_samples_leaf || sum(right_mask) < tree.min_samples_leaf
                continue
            end
            
            impurity_decrease = compute_impurity_decrease(tree, y, y[left_mask], y[right_mask])
            
            if impurity_decrease > best_impurity_decrease
                best_impurity_decrease = impurity_decrease
                best_feature = feature
                best_threshold = threshold
            end
        end
    end
    
    if best_impurity_decrease < tree.min_impurity_decrease
        return nothing, nothing
    end
    
    return best_feature, best_threshold
end

function compute_impurity_decrease(tree::DecisionTreeRegressor, y_parent::AbstractVector, y_left::AbstractVector, y_right::AbstractVector)
    n = length(y_parent)
    n_left = length(y_left)
    n_right = length(y_right)
    
    if tree.criterion == "squared_error"
        impurity_parent = var(y_parent)
        impurity_left = var(y_left)
        impurity_right = var(y_right)
    elseif tree.criterion == "friedman_mse"
        # Implement Friedman MSE
        impurity_parent = var(y_parent)
        impurity_left = var(y_left)
        impurity_right = var(y_right)
        # Add Friedman's improvement score
    elseif tree.criterion == "absolute_error"
        impurity_parent = mean(abs.(y_parent .- median(y_parent)))
        impurity_left = mean(abs.(y_left .- median(y_left)))
        impurity_right = mean(abs.(y_right .- median(y_right)))
    elseif tree.criterion == "poisson"
        # Implement Poisson deviance
        impurity_parent = poisson_deviance(y_parent)
        impurity_left = poisson_deviance(y_left)
        impurity_right = poisson_deviance(y_right)
    else
        throw(ArgumentError("Unknown criterion: $(tree.criterion)"))
    end
    
    return impurity_parent - (n_left / n) * impurity_left - (n_right / n) * impurity_right
end

function create_leaf(y::AbstractVector)
    node = RegressorNode()
    node.is_leaf = true
    node.value = mean(y)
    return node
end

function predict_sample(tree::DecisionTreeRegressor, node::RegressorNode, x::AbstractVector)
    if node.is_leaf
        return node.value
    end
    
    if x[node.feature_index] <= node.threshold
        return predict_sample(tree, node.left, x)
    else
        return predict_sample(tree, node.right, x)
    end
end

function compute_feature_importances(tree::DecisionTreeRegressor)
    importances = zeros(tree.n_features_)
    
    function traverse(node::RegressorNode, depth::Int)
        if node === nothing || node.is_leaf
            return
        end
        
        importances[node.feature_index] += 1 / depth
        
        traverse(node.left, depth + 1)
        traverse(node.right, depth + 1)
    end
    
    traverse(tree.root, 1)
    
    return importances ./ sum(importances)
end

function count_leaves(node::Union{RegressorNode, Nothing})
    if node === nothing
        return 0
    elseif node.is_leaf
        return 1
    else
        return count_leaves(node.left) + count_leaves(node.right)
    end
end

function poisson_deviance(y::AbstractVector)
    y_pred = mean(y)
    return 2 * sum(y .* log.(y ./ y_pred) .- (y .- y_pred))
end

# Additional methods to match sklearn's API
function get_depth(tree::DecisionTreeRegressor)
    function max_depth(node::Union{RegressorNode, Nothing})
        if node === nothing
            return 0
        elseif node.is_leaf
            return 1
        else
            return 1 + max(max_depth(node.left), max_depth(node.right))
        end
    end
    
    return max_depth(tree.root)
end

function get_n_leaves(tree::DecisionTreeRegressor)
    return count_leaves(tree.root)
end

function apply(tree::DecisionTreeRegressor, X::AbstractMatrix)
    function find_leaf_index(node::RegressorNode, x::AbstractVector, index::Int)
        if node.is_leaf
            return index
        end
        
        if x[node.feature_index] <= node.threshold
            return find_leaf_index(node.left, x, 2 * index)
        else
            return find_leaf_index(node.right, x, 2 * index + 1)
        end
    end
    
    return [find_leaf_index(tree.root, x, 1) for x in eachrow(X)]
end

function decision_path(tree::DecisionTreeRegressor, X::AbstractMatrix)
    function traverse(node::RegressorNode, x::AbstractVector, path::Vector{Int})
        push!(path, 1)
        
        if node.is_leaf
            return path
        end
        
        if x[node.feature_index] <= node.threshold
            traverse(node.left, x, path)
        else
            traverse(node.right, x, path)
        end
        
        return path
    end
    
    paths = [traverse(tree.root, x, Int[]) for x in eachrow(X)]
    max_path_length = maximum(length.(paths))
    
    indicator = zeros(Int, length(paths), max_path_length)
    for (i, path) in enumerate(paths)
        indicator[i, 1:length(path)] = path
    end
    
    return indicator
end

function Base.show(io::IO, tree::DecisionTreeRegressor)
    println(io, "DecisionTreeRegressor(")
    println(io, "  criterion=$(tree.criterion),")
    println(io, "  splitter=$(tree.splitter),")
    println(io, "  max_depth=$(tree.max_depth),")
    println(io, "  min_samples_split=$(tree.min_samples_split),")
    println(io, "  min_samples_leaf=$(tree.min_samples_leaf),")
    println(io, "  min_weight_fraction_leaf=$(tree.min_weight_fraction_leaf),")
    println(io, "  max_features=$(tree.max_features),")
    println(io, "  random_state=$(tree.random_state),")
    println(io, "  max_leaf_nodes=$(tree.max_leaf_nodes),")
    println(io, "  min_impurity_decrease=$(tree.min_impurity_decrease),")
    println(io, "  ccp_alpha=$(tree.ccp_alpha),")
    println(io, "  fitted=$(tree.fitted)")
    print(io, ")")
end