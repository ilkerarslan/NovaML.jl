using Statistics
using LinearAlgebra

function f1_score(y_true, y_pred; 
                  labels=nothing, 
                  pos_label=1, 
                  average="binary", 
                  sample_weight=nothing, 
                  zero_division="warn")
    
    if labels === nothing
        labels = sort(unique(vcat(y_true, y_pred)))
    end
    
    if average == "binary"
        if length(labels) != 2
            throw(ArgumentError("average='binary' requires binary targets"))
        end
        pos_label_idx = findfirst(==(pos_label), labels)
        if pos_label_idx === nothing
            throw(ArgumentError("pos_label not in labels"))
        end
        labels = [labels[pos_label_idx]]
    end
    
    f1_scores = Float64[]
    
    for label in labels
        true_positives = sum((y_true .== label) .& (y_pred .== label))
        false_positives = sum((y_true .!= label) .& (y_pred .== label))
        false_negatives = sum((y_true .== label) .& (y_pred .!= label))
        
        if true_positives == 0 && false_positives == 0 && false_negatives == 0
            if zero_division == "warn"
                @warn "Zero division encountered in F1 score"
                push!(f1_scores, 0.0)
            elseif zero_division isa Number
                push!(f1_scores, float(zero_division))
            else
                push!(f1_scores, NaN)
            end
        else
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1 = 2 * (precision * recall) / (precision + recall)
            push!(f1_scores, f1)
        end
    end
    
    if sample_weight !== nothing
        f1_scores .*= sample_weight
    end
    
    if average == "micro"
        # For micro-average F1, we need to calculate precision and recall globally
        true_positives = sum((y_true .== y_pred) .& (y_true .∈ Ref(Set(labels))))
        false_positives = sum((y_true .!= y_pred) .& (y_pred .∈ Ref(Set(labels))))
        false_negatives = sum((y_true .!= y_pred) .& (y_true .∈ Ref(Set(labels))))
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        return 2 * (precision * recall) / (precision + recall)
    elseif average == "macro"
        return mean(f1_scores)
    elseif average == "weighted"
        weights = [sum(y_true .== label) for label in labels]
        return dot(f1_scores, weights) / sum(weights)
    elseif average == "samples"
        throw(ArgumentError("average='samples' is not supported for binary or multiclass classification"))
    elseif average === nothing || average == "binary"
        return f1_scores
    else
        throw(ArgumentError("Unsupported average type"))
    end
end