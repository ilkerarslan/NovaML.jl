using Random

import ...Nova: AbstractModel

export DecisionTreeClassifier

mutable struct Node
    feature_index::Union{Int, Nothing}
    threshold::Union{Float64, Nothing}
    left::Union{Node, Nothing}
    right::Union{Node, Nothing}
    is_leaf::Bool
    class::Union{Int, Nothing}

    Node() = new(nothing, nothing, nothing, nothing, false, nothing)
end

mutable struct DecisionTreeClassifier <: AbstractModel
    max_depth::Union{Int, Nothing}
    min_samples_split::Int
    min_samples_leaf::Int
    random_state::Union{Int, Nothing}
    root::Union{Node, Nothing}
    n_classes::Int
    classes::Vector
    fitted::Bool

    function DecisionTreeClassifier(;
        max_depth=nothing,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=nothing
    )
        new(max_depth, min_samples_split, min_samples_leaf, random_state, nothing, 0, [], false)
    end
end

function (tree::DecisionTreeClassifier)(X::AbstractMatrix, y::AbstractVector)
    tree.classes = unique(y)
    tree.n_classes = length(tree.classes)
    
    if tree.random_state !== nothing
        Random.seed!(tree.random_state)
    end
    
    tree.root = grow_tree(tree, X, y, 0)
    tree.fitted = true
    return tree
end

function (tree::DecisionTreeClassifier)(X::AbstractMatrix)
    if !tree.fitted
        throw(ErrorException("This DecisionTreeClassifier instance is not fitted yet. Call the model with training data before using it for predictions."))
    end
    return [predict_sample(tree, tree.root, x) for x in eachrow(X)]
end

function grow_tree(tree::DecisionTreeClassifier, X::AbstractMatrix, y::AbstractVector, depth::Int)
    n_samples, n_features = size(X)
    
    # Check stopping criteria
    if (tree.max_depth !== nothing && depth >= tree.max_depth) ||
       n_samples < tree.min_samples_split ||
       n_samples < 2 * tree.min_samples_leaf ||
       length(unique(y)) == 1
        return create_leaf(y, tree.classes)
    end
    
    # Find the best split
    best_feature, best_threshold = find_best_split(X, y, tree.min_samples_leaf)
    
    if best_feature === nothing
        return create_leaf(y, tree.classes)
    end
    
    # Split the data
    left_mask = X[:, best_feature] .<= best_threshold
    right_mask = .!left_mask
    
    # Create the node
    node = Node()
    node.feature_index = best_feature
    node.threshold = best_threshold
    
    # Recursively build the left and right subtrees
    node.left = grow_tree(tree, X[left_mask, :], y[left_mask], depth + 1)
    node.right = grow_tree(tree, X[right_mask, :], y[right_mask], depth + 1)
    
    return node
end

function find_best_split(X::AbstractMatrix, y::AbstractVector, min_samples_leaf::Int)
    n_samples, n_features = size(X)
    best_gini = Inf
    best_feature = nothing
    best_threshold = nothing
    
    for feature in 1:n_features
        feature_values = unique(X[:, feature])
        
        for threshold in feature_values
            left_mask = X[:, feature] .<= threshold
            right_mask = .!left_mask
            
            if sum(left_mask) < min_samples_leaf || sum(right_mask) < min_samples_leaf
                continue
            end
            
            gini = gini_impurity(y[left_mask]) * sum(left_mask) / n_samples +
                   gini_impurity(y[right_mask]) * sum(right_mask) / n_samples
            
            if gini < best_gini
                best_gini = gini
                best_feature = feature
                best_threshold = threshold
            end
        end
    end
    
    return best_feature, best_threshold
end

function gini_impurity(y::AbstractVector)
    n = length(y)
    return 1.0 - sum((count(y .== c) / n)^2 for c in unique(y))
end

function create_leaf(y::AbstractVector, classes::Vector)
    node = Node()
    node.is_leaf = true
    node.class = classes[argmax([count(y .== c) for c in classes])]
    return node
end

function predict_sample(tree::DecisionTreeClassifier, node::Node, x::AbstractVector)
    if node.is_leaf
        return node.class
    end
    
    if x[node.feature_index] <= node.threshold
        return predict_sample(tree, node.left, x)
    else
        return predict_sample(tree, node.right, x)
    end
end

function Base.show(io::IO, tree::DecisionTreeClassifier)
    println(io, "DecisionTreeClassifier(")
    println(io, "  max_depth=$(tree.max_depth),")
    println(io, "  min_samples_split=$(tree.min_samples_split),")
    println(io, "  min_samples_leaf=$(tree.min_samples_leaf),")
    println(io, "  random_state=$(tree.random_state),")
    println(io, "  fitted=$(tree.fitted)")
    print(io, ")")
end