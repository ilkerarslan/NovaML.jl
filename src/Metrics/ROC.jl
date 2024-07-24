# File: src/Metrics/ROC.jl
using Statistics

function roc_curve(y_true::AbstractVector, y_score::AbstractVector; 
                   pos_label=nothing, sample_weight=nothing, drop_intermediate=true)
    if pos_label === nothing
        pos_label = maximum(y_true)
    end
    
    # Ensure binary classification
    if !all(y ∈ [pos_label, pos_label == 1 ? 0 : 1] for y in y_true)
        throw(ArgumentError("y_true should be a binary vector"))
    end
    
    # Sort scores and corresponding truth values
    desc_score_indices = sortperm(y_score, rev=true)
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    if sample_weight !== nothing
        sample_weight = sample_weight[desc_score_indices]
    end
    
    # Compute true positive rates and false positive rates
    distinct_value_indices = findall(i -> i == 1 || y_score[i] != y_score[i-1], 1:length(y_score))
    threshold_idxs = [1; distinct_value_indices; length(y_score)]
    
    tps = cumsum(y_true .== pos_label)
    fps = cumsum(y_true .!= pos_label)
    
    if sample_weight !== nothing
        tps = [sum((y_true .== pos_label)[1:i] .* sample_weight[1:i]) for i in threshold_idxs]
        fps = [sum((y_true .!= pos_label)[1:i] .* sample_weight[1:i]) for i in threshold_idxs]
    else
        tps = tps[threshold_idxs]
        fps = fps[threshold_idxs]
    end
    
    tps = [0; tps]
    fps = [0; fps]
    
    if drop_intermediate && length(threshold_idxs) > 2
        optimal_idxs = [1]
        last_fps = fps[1]
        last_tps = tps[1]
        for i in 2:length(threshold_idxs)
            if fps[i] != last_fps || tps[i] != last_tps
                push!(optimal_idxs, i)
                last_fps = fps[i]
                last_tps = tps[i]
            end
        end
        threshold_idxs = threshold_idxs[optimal_idxs]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
    end
    
    if tps[end] <= 0
        fpr = [Inf; fps / fps[end]]
        tpr = [Inf; tps / tps[end]]
    else
        fpr = fps / fps[end]
        tpr = tps / tps[end]
    end
    
    thresholds = [Inf; y_score[threshold_idxs[2:end]]]
    
    return fpr, tpr, thresholds
end

function auc(x::AbstractVector, y::AbstractVector)
    if !issorted(x) && !issorted(x, rev=true)
        throw(ArgumentError("x must be monotonic"))
    end
    
    direction = issorted(x) ? 1 : -1
    dx = direction .* diff(x)
    return sum((y[1:end-1] .+ y[2:end]) ./ 2 .* dx)
end


# File: src/Metrics/ROC.jl

function roc_auc_score(y_true::AbstractVector, y_score::AbstractVecOrMat;
                       average::Union{String, Nothing}="macro",
                       multi_class::String="raise",
                       labels::Union{AbstractVector, Nothing}=nothing)

    if y_score isa AbstractVector || size(y_score, 2) == 1
    # Binary classification
    y_score_vec = y_score isa AbstractVector ? y_score : vec(y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score_vec)
    return auc(fpr, tpr)
    end

    n_classes = size(y_score, 2)

    if multi_class == "raise" && n_classes > 2
    throw(ArgumentError("multi_class must be in ['ovr', 'ovo'] for multiclass problems"))
    end

    if labels === nothing
    labels = unique(sort(y_true))
    end

    if n_classes == 2
    # Binary classification with probability scores for both classes
    fpr, tpr, _ = roc_curve(y_true, y_score[:, 2])
    return auc(fpr, tpr)
    end

    if multi_class == "ovr"
    # One-vs-Rest
    auc_scores = Float64[]
    for (i, label) in enumerate(labels)
    y_true_binary = (y_true .== label)
    fpr, tpr, _ = roc_curve(y_true_binary, y_score[:, i])
    push!(auc_scores, auc(fpr, tpr))
    end
    elseif multi_class == "ovo"
    # One-vs-One
    auc_scores = Float64[]
    n_classes = length(labels)
    for i in 1:n_classes
    for j in (i+1):n_classes
    mask = (y_true .== labels[i]) .| (y_true .== labels[j])
    if !any(mask)
     continue
    end
    y_true_binary = (y_true[mask] .== labels[i])
    y_score_binary = y_score[mask, i] ./ (y_score[mask, i] .+ y_score[mask, j])
    fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
    push!(auc_scores, auc(fpr, tpr))
    end
    end
    else
    throw(ArgumentError("multi_class must be in ['ovr', 'ovo']"))
    end

    if average == "macro"
    return mean(auc_scores)
    elseif average == "weighted"
    class_weights = [count(==(label), y_true) for label in labels]
    return sum(auc_scores .* (class_weights ./ sum(class_weights)))
    elseif average === nothing
    return auc_scores
    else
    throw(ArgumentError("average must be in ['macro', 'weighted'] or nothing"))
    end
end

# Helper function for binary classification with probability scores
function roc_auc_score(y_true::AbstractVector, y_score::AbstractVector)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)
end