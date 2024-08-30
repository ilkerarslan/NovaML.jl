using LinearAlgebra
using Statistics

function matthews_corrcoef(y_true, y_pred; sample_weight=nothing)
    if length(y_true) != length(y_pred)
        throw(ArgumentError("y_true and y_pred must have the same length"))
    end
    
    # Get unique labels
    labels = sort(unique(vcat(y_true, y_pred)))
    
    if length(labels) == 1
        return 0.0
    end
    
    if length(labels) == 2
        return binary_matthews_corrcoef(y_true, y_pred, labels, sample_weight)
    else
        return multiclass_matthews_corrcoef(y_true, y_pred, labels, sample_weight)
    end
end

function binary_matthews_corrcoef(y_true, y_pred, labels, sample_weight)
    # Compute confusion matrix
    tp = sum((y_true .== y_pred) .& (y_true .== labels[2]))
    tn = sum((y_true .== y_pred) .& (y_true .== labels[1]))
    fp = sum((y_true .!= y_pred) .& (y_pred .== labels[2]))
    fn = sum((y_true .!= y_pred) .& (y_pred .== labels[1]))
    
    if sample_weight !== nothing
        tp = sum(sample_weight[(y_true .== y_pred) .& (y_true .== labels[2])])
        tn = sum(sample_weight[(y_true .== y_pred) .& (y_true .== labels[1])])
        fp = sum(sample_weight[(y_true .!= y_pred) .& (y_pred .== labels[2])])
        fn = sum(sample_weight[(y_true .!= y_pred) .& (y_pred .== labels[1])])
    end
    
    numerator = (tp * tn - fp * fn)
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0
        return 0.0
    else
        return numerator / denominator
    end
end

function multiclass_matthews_corrcoef(y_true, y_pred, labels, sample_weight)
    # Compute confusion matrix
    n_classes = length(labels)
    confusion_matrix = zeros(n_classes, n_classes)
    
    for (i, true_label) in enumerate(labels)
        for (j, pred_label) in enumerate(labels)
            if sample_weight === nothing
                confusion_matrix[i, j] = sum((y_true .== true_label) .& (y_pred .== pred_label))
            else
                confusion_matrix[i, j] = sum(sample_weight[(y_true .== true_label) .& (y_pred .== pred_label)])
            end
        end
    end
    
    t = sum(confusion_matrix)
    
    # Calculate sums for each row and column
    sum_over_rows = sum(confusion_matrix, dims=2)
    sum_over_cols = sum(confusion_matrix, dims=1)
    
    numerator = sum(confusion_matrix .* confusion_matrix) - sum(sum_over_rows .* sum_over_cols)
    denominator = sqrt((t^2 - sum(sum_over_rows.^2)) * (t^2 - sum(sum_over_cols.^2)))
    
    if denominator == 0
        return 0.0
    else
        return numerator / denominator
    end
end