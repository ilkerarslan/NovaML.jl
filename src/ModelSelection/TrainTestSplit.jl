using Random, StatsBase
using DataFrames

function train_test_split(df::DataFrame; test_size::Union{Float64,Int}=0.25,
    shuffle::Bool=true, random_state::Union{Int,Nothing}=nothing,
    stratify::Union{Symbol,AbstractVector,Nothing}=nothing)

    if random_state !== nothing
        Random.seed!(random_state)
    end

    n = nrow(df)

    n_test = if test_size isa Float64
        @assert 0 < test_size < 1 "test_size as fraction must be between 0 and 1"
        round(Int, test_size * n)
    else
        @assert 0 < test_size < n "test_size as integer must be between 0 and number of rows"
        test_size
    end

    if stratify !== nothing
        strat_values = if stratify isa Symbol
            if !(stratify in propertynames(df))
                throw(ArgumentError("Stratification column $stratify not found in DataFrame"))
            end
            df[!, stratify]
        else
            if length(stratify) != n
                throw(ArgumentError("Stratification vector length must match DataFrame length"))
            end
            stratify
        end

        class_counts = countmap(strat_values)
        classes = collect(keys(class_counts))

        train_indices = Int[]
        test_indices = Int[]

        for class in classes
            class_indices = findall(x -> x == class, strat_values)
            n_class = length(class_indices)
            n_test_class = round(Int, n_test * n_class / n)

            if shuffle
                class_indices = Random.shuffle(class_indices)
            end

            append!(test_indices, class_indices[1:n_test_class])
            append!(train_indices, class_indices[n_test_class+1:end])
        end

        if shuffle
            Random.shuffle!(train_indices)
            Random.shuffle!(test_indices)
        end
    else
        indices = 1:n
        if shuffle
            indices = Random.shuffle(indices)
        end
        test_indices = indices[1:n_test]
        train_indices = indices[n_test+1:end]
    end

    df_train = df[train_indices, :]
    df_test = df[test_indices, :]

    return df_train, df_test
end

function train_test_split(X::Union{AbstractMatrix,AbstractVector}, y::AbstractVector;
    test_size=0.25, shuffle=true, random_state=nothing, stratify=nothing)
    if random_state !== nothing
        Random.seed!(random_state)
    end

    n = size(X, 1)
    test_size = round(Int, test_size * n)

    if stratify !== nothing
        if length(stratify) != n
            throw(ArgumentError("stratify must have the same length as X and y"))
        end

        class_counts = countmap(stratify)
        classes = collect(keys(class_counts))

        train_indices = Int[]
        test_indices = []

        for class in classes
            class_indices = findall(x -> x == class, stratify)
            n_class = length(class_indices)
            n_test = round(Int, test_size * n_class / n)

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
    else
        indices = 1:n
        if shuffle
            indices = Random.shuffle(indices)
        end
        test_indices = indices[1:test_size]
        train_indices = indices[test_size+1:end]
    end

    if X isa AbstractMatrix
        Xtrn, Xtst = X[train_indices, :], X[test_indices, :]
    else
        Xtrn, Xtst = X[train_indices], X[test_indices]
    end

    ytrn, ytst = y[train_indices], y[test_indices]

    return Xtrn, Xtst, ytrn, ytst
end