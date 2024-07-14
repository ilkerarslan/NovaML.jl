using Random

export StratifiedKFold

struct StratifiedKFold
    n_splits::Int
    shuffle::Bool
    rng::Random.AbstractRNG

    function StratifiedKFold(; n_splits::Int=5, shuffle::Bool=false, random_state::Union{Int, Nothing}=nothing)
        if n_splits < 2
            throw(ArgumentError("n_splits must be ≥ 2"))
        end
        rng = isnothing(random_state) ? Random.GLOBAL_RNG : Random.MersenneTwister(random_state)
        new(n_splits, shuffle, rng)
    end
end

function (kfold::StratifiedKFold)(y::AbstractVector)
    n_samples = length(y)
    indices = collect(1:n_samples)
    
    # Get unique classes and their indices
    unique_classes = unique(y)
    class_indices = Dict(c => findall(x -> x == c, y) for c in unique_classes)

    # Shuffle indices if required
    if kfold.shuffle
        for indices in values(class_indices)
            shuffle!(kfold.rng, indices)
        end
    end

    # Calculate fold sizes for each class
    fold_sizes = Dict(c => div.(fill(length(indices), kfold.n_splits), kfold.n_splits) for (c, indices) in class_indices)
    
    # Distribute remaining samples
    for (c, sizes) in fold_sizes
        remainder = length(class_indices[c]) - sum(sizes)
        for i in 1:remainder
            sizes[i] += 1
        end
    end

    # Generate folds
    folds = [Int[] for _ in 1:kfold.n_splits]
    for (c, indices) in class_indices
        start = 1
        for (fold, size) in enumerate(fold_sizes[c])
            finish = start + size - 1
            append!(folds[fold], indices[start:finish])
            start = finish + 1
        end
    end

    # Create iterator
    return ((setdiff(indices, test), test) for test in folds)
end

Base.iterate(kfold::StratifiedKFold, state=1) = state > kfold.n_splits ? nothing : (state, state + 1)
Base.length(kfold::StratifiedKFold) = kfold.n_splits