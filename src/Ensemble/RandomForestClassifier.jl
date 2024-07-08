module RandomForestClassifierModel

using Random
using Statistics: mean

import ...Nova: AbstractModel
import ...Nova.Tree: DecisionTreeClassifier

export RandomForestClassifier

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
            false
        )
    end
end

function (forest::RandomForestClassifier)(X::AbstractMatrix, y::AbstractVector)
    n_samples, n_features = size(X)
    forest.classes = unique(y)
    forest.n_classes = length(forest.classes)

    if forest.random_state !== nothing
        Random.seed!(forest.random_state)
    end

    # Determine the number of features to consider for each split
    if forest.max_features === nothing
        max_features = n_features
    elseif isa(forest.max_features, Int)
        max_features = forest.max_features
    elseif isa(forest.max_features, Float64)
        max_features = round(Int, forest.max_features * n_features)
    elseif forest.max_features == "sqrt"
        max_features = round(Int, sqrt(n_features))
    elseif forest.max_features == "log2"
        max_features = round(Int, log2(n_features))
    else
        throw(ArgumentError("Invalid max_features parameter"))
    end

    forest.trees = []

    for _ in 1:forest.n_estimators
        tree = DecisionTreeClassifier(
            max_depth=forest.max_depth,
            min_samples_split=forest.min_samples_split,
            min_samples_leaf=forest.min_samples_leaf,
            random_state=forest.random_state !== nothing ? rand(1:10000) : nothing
        )

        if forest.bootstrap
            indices = rand(1:n_samples, n_samples)
            X_bootstrap = X[indices, :]
            y_bootstrap = y[indices]
        else
            X_bootstrap = X
            y_bootstrap = y
        end

        # Randomly select features for this tree
        feature_indices = randperm(n_features)[1:max_features]
        X_subset = X_bootstrap[:, feature_indices]

        tree(X_subset, y_bootstrap)
        push!(forest.trees, tree)
    end

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

end # of module RandomForestClassifierModel