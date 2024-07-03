module ModelSelection

using Random, StatsBase

export train_test_split

"""
    train_test_split(X, y; test_size=0.25, shuffle=true, random_state=nothing, stratify=nothing)

Split arrays or matrices into random train and test subsets.

# Arguments
- `X`: Array or matrix of features
- `y`: array of labels
- `test_size`: Proportion of the dataset to include in the test split (default: 0.25)
- `shuffle`: Whether to shuffle the data before splitting (default: true)
- `random_state`: Random seed for reproducability (default: nothing)
- `stratify`: Array used for stratified sampling (default: nothing)

# Returns
- `Xtrn`, `Xtst`, `ytrn`, `ytst`
"""
function train_test_split(X, y; test_size=0.25, shuffle=true, random_state=nothing, stratify=nothing)
    if random_state !== nothing
        Random.seed!(random_state)
    end

    n = size(X, 1)
    test_size = round(Int, test_size*n)

    if stratify === nothing
        indices = 1:n
        if shuffle
            indices = Random.shuffle(indices)
        end
        test_indices = indices[1:test_size]
        train_indices = indices[test_size+1:end]
    else
        if length(stratify) != n
            throw(ArgumentError("stratify must have the same length as X and y"))
        end

        # Get unique classes and their frequencies
        class_counts = countmap(stratify)
        classes = collect(keys(class_counts))

        train_indices = Int[]
        test_indices = []

        for class in classes
            class_indices = findall(x -> x == class, stratify)
            n_class = length(class_indices)
            n_test = round(Int, test_size*n_class / n)

            if shuffle
                class_indices = Random.shuffle(class_indices)
            end

            append!(test_indices, class_indices[1:n_test])
            append!(train_indices, class_indices[n_test+1:end])
        end

        if shuffle
            Random.shuffle!(train_indices)
            Random.shuffle!(test_indices)
        end
    end

    if X isa AbstractMatrix
        Xtrn, Xtst = X[train_indices, :], X[test_indices, :]
    else
        Xtrn, Xtst = X[train_indices], X[test_indices]
    end

    ytrn, ytst = y[train_indices], y[test_indices]

    return Xtrn, Xtst, ytrn, ytst
end


end # of module ModelSelection