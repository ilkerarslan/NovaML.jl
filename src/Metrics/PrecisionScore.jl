using Statistics
using LinearAlgebra

function precision_score(y_true, y_pred; 
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
    
    precisions = Float64[]
    
    for label in labels
        true_positives = sum((y_true .== label) .& (y_pred .== label))
        predicted_positives = sum(y_pred .== label)
        
        if predicted_positives == 0
            if zero_division == "warn"
                @warn "Zero division encountered in precision score"
                push!(precisions, 0.0)
            elseif zero_division isa Number
                push!(precisions, float(zero_division))
            else
                push!(precisions, NaN)
            end
        else
            push!(precisions, true_positives / predicted_positives)
        end
    end
    
    if sample_weight !== nothing
        precisions .*= sample_weight
    end
    
    if average == "micro"
        return mean(precisions)
    elseif average == "macro"
        return mean(precisions)
    elseif average == "weighted"
        weights = [sum(y_true .== label) for label in labels]
        return dot(precisions, weights) / sum(weights)
    elseif average === nothing || average == "binary"
        return precisions
    else
        throw(ArgumentError("Unsupported average type"))
    end
end