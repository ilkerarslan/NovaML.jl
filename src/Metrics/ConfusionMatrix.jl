using StatsBase
    
"""
    confusion_matrix(y_true, y_pred; labels=nothing, sample_weight=nothing, normalize=nothing)

Compute confusion matrix to evaluate the accuracy of a classification.

Parameters:
- `y_true`: Vector of true labels
- `y_pred`: Vector of predicted labels
- `labels`: Optional vector of label values to include in the matrix
- `sample_weight`: Optional vector of sample weights
- `normalize`: Optional normalization strategy ('true', 'pred', 'all', or nothing)

Returns:
- Confusion matrix as a 2D array
"""
function confusion_matrix(y_true, y_pred; 
                          labels=nothing, 
                          sample_weight=nothing, 
                          normalize=nothing)
    if labels === nothing
        labels = sort(unique(vcat(y_true, y_pred)))
    end
    
    n_labels = length(labels)
    label_to_ind = Dict(label => i for (i, label) in enumerate(labels))
    
    # Initialize confusion matrix
    cm = zeros(Int, n_labels, n_labels)
    
    # Fill confusion matrix
    for (yt, yp) in zip(y_true, y_pred)
        if yt in labels && yp in labels
            i, j = label_to_ind[yt], label_to_ind[yp]
            cm[i, j] += sample_weight !== nothing ? sample_weight[i] : 1
        end
    end
    
    # Normalize if required
    if normalize !== nothing
        if normalize == true
            cm = cm ./ sum(cm, dims=2)
        elseif normalize == :pred
            cm = cm ./ sum(cm, dims=1)
        elseif normalize == :all
            cm = cm ./ sum(cm)
        else
            throw(ArgumentError("Invalid normalize option. Use true, :pred, :all, or nothing."))
        end
        
        # Replace NaN with 0 and convert to Float64
        cm = replace(cm, NaN => 0.0)
    end
    
    return cm
end

# Helper function to display confusion matrix
function display_confusion_matrix(cm)
    n = size(cm, 1)
    labels = string.(1:n)
    
    # Print header
    print("   ")
    for label in labels
        print(lpad(label, 5))
    end
    println("\n   ", "-----"^n)
    
    # Print rows
    for i in 1:n
        print(labels[i], " |")
        for j in 1:n
            print(lpad(string(round(cm[i,j], digits=2)), 5))
        end
        println()
    end
end