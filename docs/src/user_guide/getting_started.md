# Getting Started

## Installation

You can install NovaML.jl using Julia's package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

`pkg> add NovaML`

## Usage

The most prominent feature of NovaML is using functors (callable objects) to keep parameters as well as training and prediction. Assume ``model`` represents a supervised algorithm. The struct ``model`` keeps learned parameters and hyperparameters. It also behave as a function. 

* `model(X, y)` trains the model. 
* `model(Xnew)` calculates the predictions for `Xnew`. 

Here's a quick example of how to use NovaML.jl for a binary classification task:

```julia
# Import the Iris dataset from NovaML's Datasets module
using NovaML.Datasets
X, y = load_iris(return_X_y=true)

# Split the data into training and test sets
# 80% for training, 20% for testing
using NovaML.ModelSelection
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2)

# Import the StandardScaler for feature scaling
using NovaML.PreProcessing
scaler = StandardScaler()
scaler.fitted # false - the scaler is not yet fitted to any data

# Fit the scaler to the training data and transform it
# NovaML uses a functor approach, so calling scaler(Xtrn) both fits and transforms
Xtrnstd = scaler(Xtrn) 

# Transform the test data using the fitted scaler
Xtststd = scaler(Xtst)

# Import LogisticRegression from LinearModel module
using NovaML.LinearModel

# Create a LogisticRegression model with learning rate 0.1 and 100 iterations
lr = LogisticRegression(η=0.1, num_iter=100)

# Import OneVsRestClassifier for multi-class classification
using NovaML.MultiClass
# Wrap the LogisticRegression model in a OneVsRestClassifier for multi-class support
ovr = OneVsRestClassifier(lr)

# Fit the OneVsRestClassifier model to the standardized training data
# NovaML uses functors, so ovr(Xtrnstd, ytrn) fits the model
ovr(Xtrnstd, ytrn)

# Make predictions on training and test data
# Calling ovr(X) makes predictions using the fitted model
ŷtrn = ovr(Xtrnstd)
ŷtst = ovr(Xtststd)

# Import accuracy_score metric for model evaluation
using NovaML.Metrics

# Calculate accuracy for training and test sets
acc_trn = accuracy_score(ytrn, ŷtrn);
acc_tst = accuracy_score(ytst, ŷtst);

# Print the results
println("Training accuracy: $acc_trn")
println("Test accuracy: $acc_tst")
# Output:
# Training accuracy: 0.9833333333333333
# Test accuracy: 0.9666666666666667
```