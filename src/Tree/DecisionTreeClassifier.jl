"""
Nova.Tree module containing the DecisionTreeClassifier implementation.
"""
module DecisionTreeClassifierModel

using Random

import ...Nova: AbstractModel

import Base: show
import Random: shuffle
import Statistics: mean

export DecisionTreeClassifier

"""
    Node

Represents a node in the decision tree.

# Fields
- `feature::Union{Int, Nothing}`: The feature index used for splitting at this node.
- `threshold::Union{Float64, Nothing}`: The threshold value for the split.
- `left::Union{Node, Nothing}`: The left child node.
- `right::Union{Node, Nothing}`: The right child node.
- `value::Union{Vector{Float64}, Nothing}`: The class probabilities if this is a leaf node.
"""
mutable struct Node
    feature::Union{Int, Nothing}
    threshold::Union{Float64, Nothing}
    left::Union{Node, Nothing}
    right::Union{Node, Nothing}
    value::Union{Vector{Float64}, Nothing}
end

# Default constructor for Node
Node() = Node(nothing, nothing, nothing, nothing, nothing)

"""
    DecisionTreeClassifier <: AbstractModel

A decision tree classifier implementation.

# Fields
- `max_depth::Union{Int, Nothing}`: The maximum depth of the tree.
- `min_samples_split::Int`: The minimum number of samples required to split an internal node.
- `min_samples_leaf::Int`: The minimum number of samples required to be at a leaf node.
- `random_state::Union{Int, Nothing}`: Seed for random number generator.
- `root::Union{Node, Nothing}`: The root node of the decision tree.
- `n_classes::Int`: The number of classes.
- `classes::Vector`: The unique class labels.
- `fitted::Bool`: Whether the model has been fitted to data.

# Constructor
    DecisionTreeClassifier(;
        max_depth=nothing,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=nothing
    )

Create a new DecisionTreeClassifier with the specified hyperparameters.
"""
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

"""
    (tree::DecisionTreeClassifier)(X::AbstractMatrix, y::AbstractVector)

Fit the decision tree classifier to the training data.

# Arguments
- `X::AbstractMatrix`: The input features.
- `y::AbstractVector`: The target values.

# Returns
- The fitted DecisionTreeClassifier object.
"""
function (tree::DecisionTreeClassifier)(X::AbstractMatrix, y::AbstractVector)
    tree.classes = unique(y)
    tree.n_classes = length(tree.classes)
    
    if tree.random_state !== nothing
        Random.seed!(tree.random_state)
    end
    
    tree.root = grow_tree(tree, X, y, 0)
    tree.fitted = true  # Mark the tree as fitted
    return tree
end

"""
    (tree::DecisionTreeClassifier)(X::AbstractMatrix)

Predict class labels for samples in X.

# Arguments
- `X::AbstractMatrix`: The input features.

# Returns
- An array of predicted class labels.

# Throws
- `ErrorException`: If the model hasn't been fitted yet.
"""
function (tree::DecisionTreeClassifier)(X::AbstractMatrix)
    if !tree.fitted
        throw(ErrorException("This DecisionTreeClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."))
    end
    return [predict_sample(tree, tree.root, x) for x in eachrow(X)]
end

"""
    grow_tree(tree::DecisionTreeClassifier, X::AbstractMatrix, y::AbstractVector, depth::Int)

Recursively grow the decision tree.

# Arguments
- `tree::DecisionTreeClassifier`: The decision tree classifier object.
- `X::AbstractMatrix`: The input features.
- `y::AbstractVector`: The target values.
- `depth::Int`: The current depth of the tree.

# Returns
- A Node object representing the current node in the tree.
"""
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
    node = Node(best_feature, best_threshold, nothing, nothing, nothing)
    
    # Recursively build the left and right subtrees
    node.left = grow_tree(tree, X[left_mask, :], y[left_mask], depth + 1)
    node.right = grow_tree(tree, X[right_mask, :], y[right_mask], depth + 1)
    
    return node
end

"""
    find_best_split(X::AbstractMatrix, y::AbstractVector, min_samples_leaf::Int)

Find the best feature and threshold for splitting the data.

# Arguments
- `X::AbstractMatrix`: The input features.
- `y::AbstractVector`: The target values.
- `min_samples_leaf::Int`: The minimum number of samples required to be at a leaf node.

# Returns
- A tuple (best_feature, best_threshold) representing the best split.
"""
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

"""
    gini_impurity(y::AbstractVector)

Calculate the Gini impurity of a set of labels.

# Arguments
- `y::AbstractVector`: The target values.

# Returns
- The Gini impurity as a Float64.
"""
function gini_impurity(y::AbstractVector)
    n = length(y)
    return 1.0 - sum((count(y .== c) / n)^2 for c in unique(y))
end

"""
    create_leaf(y::AbstractVector, classes::Vector)

Create a leaf node with class probabilities.

# Arguments
- `y::AbstractVector`: The target values.
- `classes::Vector`: The unique class labels.

# Returns
- A Node object representing a leaf node.
"""
function create_leaf(y::AbstractVector, classes::Vector)
    counts = [count(y .== c) for c in classes]
    return Node(nothing, nothing, nothing, nothing, counts ./ length(y))
end

"""
    predict_sample(tree::DecisionTreeClassifier, node::Node, x::AbstractVector)

Predict the class for a single sample.

# Arguments
- `tree::DecisionTreeClassifier`: The decision tree classifier object.
- `node::Node`: The current node in the decision tree.
- `x::AbstractVector`: The input feature vector.

# Returns
- The predicted class label.
"""
function predict_sample(tree::DecisionTreeClassifier, node::Node, x::AbstractVector)
    if node.value !== nothing
        return tree.classes[argmax(node.value)]
    end
    
    if x[node.feature] <= node.threshold
        return predict_sample(tree, node.left, x)
    else
        return predict_sample(tree, node.right, x)
    end
end

"""
    Base.show(io::IO, tree::DecisionTreeClassifier)

Custom pretty printing for DecisionTreeClassifier.

# Arguments
- `io::IO`: The I/O stream.
- `tree::DecisionTreeClassifier`: The decision tree classifier object to be printed.
"""
function Base.show(io::IO, tree::DecisionTreeClassifier)
    println(io, "DecisionTreeClassifier(")
    println(io, "  max_depth=$(tree.max_depth),")
    println(io, "  min_samples_split=$(tree.min_samples_split),")
    println(io, "  min_samples_leaf=$(tree.min_samples_leaf),")
    println(io, "  random_state=$(tree.random_state),")
    println(io, "  fitted=$(tree.fitted)")
    print(io, ")")
end

end # module DecisionTreeClassifierModel