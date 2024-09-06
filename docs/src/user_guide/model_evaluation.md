# Model Evaluation

NovaML provides a range of tools and metrics for evaluating the performance of machine learning models. This page covers the main evaluation techniques and metrics available in NovaML.

## Classification Metrics

### Accuracy Score

Accuracy is the ratio of correct predictions to total predictions.

```julia
using NovaML.Metrics

y = [0, 1, 1, 0, 1];
ŷ = [0, 1, 0, 0, 1]

accuracy = accuracy_score(y, ŷ);
println("Accuracy: $accuracy")
```

Following example load the Wisconsin Breast Cancer dataset, splits it to training and test sets and traing a logistic regression model with the training set. Then it calculates the accuracy_score for training and test sets. 

```julia
using NovaML.Datasets: load_breast_cancer
X, y = load_breast_cancer(return_X_y=true)

using NovaML.ModelSelection: train_test_split
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=1)

using NovaML.LinearModel

lr = LogisticRegression()

# train the model
lr(Xtrn, ytrn)

using NovaML.Metrics
ŷtrn, ŷtst = lr(Xtrn), lr(Xtst)
accuracy_score(ŷtrn, ytrn), accuracy_score(ŷtst, ytst)
# 
```

### Confusion Matrix

The confusion matrix provides a detailed breakdown of correct and incorrect classifications for each class.
You can use `confusion_matrix` to create the confusion matrix and `display_confusion_matrix` to display it with labels. 

```julia
using NovaML.Metrics
confmat = confusion_matrix(ytst, ŷtst)
# 2×2 Matrix{Int64}:
#  71   1
#   5  37
```

```julia
       1    2
   ----------
1 | 71.0  1.0
2 |  5.0 37.0
```

We can create a better looking confusion matrix plot using the following function:

```julia
function plot_confusion_matrix(confmat::Matrix)
    n = size(confmat, 1)
    
    heatmap(confmat, 
            c=:Blues, 
            alpha=0.3, 
            aspect_ratio=:equal, 
            size=(300, 300),
            xrotation=0,
            xticks=1:n, 
            yticks=1:n,
            xlims=(0.5, n+0.5), 
            ylims=(0.5, n+0.5),
            right_margin=5mm,
            xlabel="Predicted label",
            ylabel="True label",
            xmirror=true, 
            framestyle=:box, 
            legend=nothing)
    
    for i in 1:n, j in 1:n
        annotate!(j, i, text(string(confmat[i,j]), :center, 10))
    end
    
    plot!(yflip=true)
    
    display(current())
end
```

```julia
plot_confusion_matrix(confmat)
```

![Confusion matrix plot](images/plot_1.svg)


### Precision, Recall, and F1 Score

These metrics provide more detailed insights into model performance, especially for imbalanced datasets.

```julia
using NovaML.Metrics

precision_score(ytst, ŷtst)
# 0.9736842105263158
recall_score(ytst, ŷtst)
# 0.8809523809523809
f1_score(ytst, ŷtst)
# 0.925
```

### Matthews Correlation Coefficient

The Matthews Correlation Coefficient is a balanced measure for binary and multiclass classification problems.


### ROC Curve and AUC

For binary classification problems, you can compute the Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC).

## Regression Metrics

### Mean Absolute Error (MAE)

MAE measures the average magnitude of errors in a set of predictions, without considering their direction.

### Mean Squared Error (MSE)

MSE measures the average squared difference between the estimated values and the actual value.

### R-squared Score

R-squared (R²) provides a measure of how well observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model.

## Clustering Metrics

### Silhouette Score

The Silhouette Score is used to evaluate the quality of clusters in clustering algorithms.

## Cross-Validation

NovaML provides tools for cross-validation to assess model performance more robustly. See the Cross-Validation section for more details.

