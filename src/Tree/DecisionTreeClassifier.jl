# File: src/Tree/DecisionTreeClassifier.jl

using Random
import ...NovaML: AbstractModel

export DecisionTreeClassifier

mutable struct Node
    feature_index::Union{Int, Nothing}
    threshold::Union{Float64, Nothing}
    left::Union{Node, Nothing}
    right::Union{Node, Nothing}
    is_leaf::Bool
    class::Union{Int, Nothing}
    class_counts::Vector{Float64}  # Changed to Float64 to support weighted counts

    Node() = new(nothing, nothing, nothing, nothing, false, nothing, Float64[])
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
    feature_importances_::Union{Vector{Float64}, Nothing}

    function DecisionTreeClassifier(;
        max_depth=nothing,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=nothing
    )
        new(max_depth, min_samples_split, min_samples_leaf, random_state, nothing, 0, [], false, nothing)
    end
end

function (tree::DecisionTreeClassifier)(X::AbstractMatrix, y::AbstractVector; sample_weight=nothing)
    tree.classes = unique(y)
    tree.n_classes = length(tree.classes)
    
    if tree.random_state !== nothing
        Random.seed!(tree.random_state)
    end
    
    tree.root = grow_tree(tree, X, y, 0, sample_weight)
    tree.feature_importances_ = compute_feature_importances(tree)
    tree.fitted = true
    return tree
end

function (tree::DecisionTreeClassifier)(X::AbstractMatrix; type=nothing)
    if !tree.fitted
        throw(ErrorException("This DecisionTreeClassifier instance is not fitted yet. Call the model with training data before using it for predictions."))
    end
    
    if type == :probs
        return predict_proba(tree, X)
    else
        return [predict_sample(tree, tree.root, x) for x in eachrow(X)]
    end
end

function predict_proba(tree::DecisionTreeClassifier, X::AbstractMatrix)
    n_samples = size(X, 1)
    probas = zeros(Float64, n_samples, tree.n_classes)
    
    for (i, x) in enumerate(eachrow(X))
        leaf_counts = get_leaf_counts(tree, tree.root, x)
        probas[i, :] = leaf_counts ./ sum(leaf_counts)
    end
    
    return probas
end

function get_leaf_counts(tree::DecisionTreeClassifier, node::Node, x::AbstractVector)
    if node.is_leaf
        return node.class_counts
    end
    
    if x[node.feature_index] <= node.threshold
        return get_leaf_counts(tree, node.left, x)
    else
        return get_leaf_counts(tree, node.right, x)
    end
end

function grow_tree(tree::DecisionTreeClassifier, X::AbstractMatrix, y::AbstractVector, depth::Int, sample_weight=nothing)
    n_samples, n_features = size(X)
    
    # Check stopping criteria
    if (tree.max_depth !== nothing && depth >= tree.max_depth) ||
       n_samples < tree.min_samples_split ||
       n_samples < 2 * tree.min_samples_leaf ||
       length(unique(y)) == 1
        return create_leaf(y, tree.classes, sample_weight)
    end
    
    # Find the best split
    best_feature, best_threshold = find_best_split(X, y, tree.min_samples_leaf, sample_weight)
    
    if best_feature === nothing
        return create_leaf(y, tree.classes, sample_weight)
    end
    
    # Split the data
    left_mask = X[:, best_feature] .<= best_threshold
    right_mask = .!left_mask
    
    # Create the node
    node = Node()
    node.feature_index = best_feature
    node.threshold = best_threshold
    
    # Recursively build the left and right subtrees
    left_weight = sample_weight !== nothing ? sample_weight[left_mask] : nothing
    right_weight = sample_weight !== nothing ? sample_weight[right_mask] : nothing
    
    node.left = grow_tree(tree, X[left_mask, :], y[left_mask], depth + 1, left_weight)
    node.right = grow_tree(tree, X[right_mask, :], y[right_mask], depth + 1, right_weight)
    
    # Combine class counts from child nodes
    node.class_counts = node.left.class_counts + node.right.class_counts
    
    return node
end

function create_leaf(y::AbstractVector, classes::Vector, sample_weight=nothing)
    node = Node()
    node.is_leaf = true
    if sample_weight === nothing
        class_counts = Float64[count(y .== c) for c in classes]
    else
        class_counts = Float64[sum(sample_weight[y .== c]) for c in classes]
    end
    node.class = classes[argmax(class_counts)]
    node.class_counts = class_counts
    return node
end

function find_best_split(X::AbstractMatrix, y::AbstractVector, min_samples_leaf::Int, sample_weight=nothing)
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
            
            gini = gini_impurity(y[left_mask], sample_weight !== nothing ? sample_weight[left_mask] : nothing) * sum(left_mask) / n_samples +
                   gini_impurity(y[right_mask], sample_weight !== nothing ? sample_weight[right_mask] : nothing) * sum(right_mask) / n_samples
            
            if gini < best_gini
                best_gini = gini
                best_feature = feature
                best_threshold = threshold
            end
        end
    end
    
    return best_feature, best_threshold
end

function gini_impurity(y::AbstractVector, sample_weight=nothing)
    if sample_weight === nothing
        n = length(y)
        return 1.0 - sum((count(y .== c) / n)^2 for c in unique(y))
    else
        total_weight = sum(sample_weight)
        return 1.0 - sum((sum(sample_weight[y .== c]) / total_weight)^2 for c in unique(y))
    end
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

function compute_feature_importances(tree::DecisionTreeClassifier)
    n_features = length(tree.root.class_counts)
    importances = zeros(Float64, n_features)
    total_samples = sum(tree.root.class_counts)

    function traverse(node::Node, current_depth::Int)
        if !node.is_leaf
            feature = node.feature_index
            left_samples = sum(node.left.class_counts)
            right_samples = sum(node.right.class_counts)
            
            importance = (gini_impurity(node.class_counts) 
                          - (left_samples / total_samples) * gini_impurity(node.left.class_counts)
                          - (right_samples / total_samples) * gini_impurity(node.right.class_counts))
            
            importances[feature] += importance
            
            traverse(node.left, current_depth + 1)
            traverse(node.right, current_depth + 1)
        end
    end

    traverse(tree.root, 1)
    
    # Normalize importances
    if sum(importances) > 0
        importances ./= sum(importances)
    end
    
    return importances
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