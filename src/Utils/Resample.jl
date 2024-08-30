using Random
using StatsBase

function resample(arrays...; replace=true, n_samples=nothing, random_state=nothing, stratify=nothing)
if length(arrays) == 0
throw(ArgumentError("At least one array must be provided"))
end
n_arrays = length(arrays)
lengths = [size(arr, 1) for arr in arrays]

if !all(l -> l == lengths[1], lengths)
    throw(ArgumentError("All arrays must have the same length"))
end

n = lengths[1]

if n_samples === nothing
    n_samples = n
end

if !replace && n_samples > n
    throw(ArgumentError("Cannot sample $n_samples samples without replacement from a set of $n items"))
end

if random_state !== nothing
    Random.seed!(random_state)
end

if stratify !== nothing
    if size(stratify, 1) != n
        throw(ArgumentError("Stratify array must have the same length as other arrays"))
    end
    
    # Stratified sampling
    classes, counts = unique(stratify), countmap(stratify)
    n_classes = length(classes)
    
    if replace
        probs = [counts[c] / n for c in classes]
        strata = rand(Categorical(probs), n_samples)
    else
        strata = vcat([fill(c, div(n_samples * counts[c], n)) for c in classes]...)
        if length(strata) < n_samples
            extra = sample(classes, n_samples - length(strata), replace=false)
            strata = vcat(strata, extra)
        end
    end
    
    indices = Int[]
    for (i, c) in enumerate(classes)
        class_indices = findall(x -> x == c, stratify)
        n_samples_class = count(x -> x == c, strata)
        append!(indices, sample(class_indices, n_samples_class, replace=replace))
    end
else
    # Simple random sampling
    indices = replace ? rand(1:n, n_samples) : sample(1:n, n_samples, replace=false)
end

# Ensure output maintains input dimensions and types
resampled = Tuple(
    if arr isa AbstractVector
        arr[indices]
    elseif arr isa AbstractMatrix
        arr[indices, :]
    else
        throw(ArgumentError("Unsupported array type: $(typeof(arr))"))
    end
    for arr in arrays
)

return resampled
end