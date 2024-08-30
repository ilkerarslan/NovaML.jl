using Statistics
using LinearAlgebra

function recall_score(y_true, y_pred; 
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
    
    recalls = Float64[]
    
    for label in labels
        true_positives = sum((y_true .== label) .& (y_pred .== label))
        actual_positives = sum(y_true .== label)
        
        if actual_positives == 0
            if zero_division == "warn"
                @warn "Zero division encountered in recall score"
                push!(recalls, 0.0)
            elseif zero_division isa Number
                push!(recalls, float(zero_division))
            else
                push!(recalls, NaN)
            end
        else
            push!(recalls, true_positives / actual_positives)
        end
    end
    
    if sample_weight !== nothing
        recalls .*= sample_weight
    end
    
    if average == "micro"
        return mean(recalls)
    elseif average == "macro"
        return mean(recalls)
    elseif average == "weighted"
        weights = [sum(y_true .== label) for label in labels]
        return dot(recalls, weights) / sum(weights)
    elseif average == "samples"
        throw(ArgumentError("average='samples' is not supported for binary or multiclass classification"))
    elseif average === nothing || average == "binary"
        return recalls
    else
        throw(ArgumentError("Unsupported average type"))
    end
end