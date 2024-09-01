using Random
using Statistics: mean

import ...NovaML: AbstractModel
import ..Tree: DecisionTreeClassifier

mutable struct RandomForestClassifier <: AbstractModel
    n_estimators::Int
    max_depth::Union{Int, Nothing}
    min_samples_split::Int
    min_samples_leaf::Int
    max_features::Union{Int, Float64, String, Nothing}
    bootstrap::Bool
    random_state::Union{Int, Nothing}
    trees::Vector{DecisionTreeClassifier}
    n_classes::Int
    classes::Vector
    fitted::Bool
    feature_importances_::Union{Vector{Float64}, Nothing}
    n_features::Int

    function RandomForestClassifier(;
        n_estimators=100,
        max_depth=nothing,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=true,
        random_state=nothing
    )
        new(
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            bootstrap,
            random_state,
            DecisionTreeClassifier[],
            0,
            [],
            false,
            nothing,
            0
        )
    end
end

function (forest::RandomForestClassifier)(X::AbstractMatrix, y::AbstractVector)
    n_samples, n_features = size(X)
    forest.n_features = n_features
    forest.classes = unique(y)
    forest.n_classes = length(forest.classes)

    if forest.random_state !== nothing
        Random.seed!(forest.random_state)
    end

    max_features = get_max_features(forest, n_features)

    forest.trees = []
    feature_importances = zeros(n_features)

    for _ in 1:forest.n_estimators
        tree = DecisionTreeClassifier(
            max_depth=forest.max_depth,
            min_samples_split=forest.min_samples_split,
            min_samples_leaf=forest.min_samples_leaf,
            random_state=forest.random_state !== nothing ? rand(1:10000) : nothing
        )

        X_bootstrap, y_bootstrap = bootstrap_sample(forest, X, y)

        # Randomly select features for this tree
        feature_indices = randperm(n_features)[1:max_features]
        X_subset = X_bootstrap[:, feature_indices]

        tree(X_subset, y_bootstrap)
        push!(forest.trees, tree)
        
        # Update feature importances
        tree_importances = calculate_tree_feature_importance(tree, feature_indices, n_features)
        feature_importances .+= tree_importances
    end

    # Average and normalize feature importances
    forest.feature_importances_ = feature_importances ./ forest.n_estimators
    forest.feature_importances_ ./= sum(forest.feature_importances_)

    forest.fitted = true
    return forest
end

function (forest::RandomForestClassifier)(X::AbstractMatrix)
    if !forest.fitted
        throw(ErrorException("This RandomForestClassifier instance is not fitted yet. Call the model with training data before using it for predictions."))
    end

    n_samples = size(X, 1)
    predictions = zeros(Int, n_samples, forest.n_estimators)

    for (i, tree) in enumerate(forest.trees)
        predictions[:, i] = tree(X)
    end

    # Majority voting
    return [forest.classes[argmax(count(==(c), row) for c in forest.classes)] for row in eachrow(predictions)]
end

function get_max_features(forest::RandomForestClassifier, n_features::Int)
    if forest.max_features === nothing
        return n_features
    elseif isa(forest.max_features, Int)
        return forest.max_features
    elseif isa(forest.max_features, Float64)
        return round(Int, forest.max_features * n_features)
    elseif forest.max_features == "sqrt"
        return round(Int, sqrt(n_features))
    elseif forest.max_features == "log2"
        return round(Int, log2(n_features))
    else
        throw(ArgumentError("Invalid max_features parameter"))
    end
end

function bootstrap_sample(forest::RandomForestClassifier, X::AbstractMatrix, y::AbstractVector)
    n_samples = size(X, 1)
    if forest.bootstrap
        indices = rand(1:n_samples, n_samples)
        return X[indices, :], y[indices]
    else
        return X, y
    end
end

function calculate_tree_feature_importance(tree::DecisionTreeClassifier, feature_indices::Vector{Int}, n_features::Int)
    importances = zeros(n_features)
    
    function traverse(node, depth=0)
        if node === nothing || node.is_leaf
            return
        end
        
        feature = feature_indices[node.feature_index]
        importances[feature] += 1 / (depth + 1)
        
        traverse(node.left, depth + 1)
        traverse(node.right, depth + 1)
    end
    
    traverse(tree.root)
    return importances
end

function Base.show(io::IO, forest::RandomForestClassifier)
    println(io, "RandomForestClassifier(")
    println(io, "  n_estimators=$(forest.n_estimators),")
    println(io, "  max_depth=$(forest.max_depth),")
    println(io, "  min_samples_split=$(forest.min_samples_split),")
    println(io, "  min_samples_leaf=$(forest.min_samples_leaf),")
    println(io, "  max_features=$(forest.max_features),")
    println(io, "  bootstrap=$(forest.bootstrap),")
    println(io, "  random_state=$(forest.random_state),")
    println(io, "  fitted=$(forest.fitted)")
    print(io, ")")
end