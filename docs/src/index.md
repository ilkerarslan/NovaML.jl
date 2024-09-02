# NovaML.jl

**⚠️ IMPORTANT NOTE: NovaML.jl is currently in alpha stage. It is under active development and may contain bugs or incomplete features. Users should exercise caution and avoid using NovaML.jl in production environments at this time. We appreciate your interest and welcome feedback and contributions to help improve the package.**

NovaML.jl aims to provide a comprehensive and user-friendly machine learning framework written in Julia. Its objective is providing a unified API for various machine learning tasks, including supervised learning, unsupervised learning, and preprocessing, feature engineering etc.

**Main objective of NovaML.jl is to increase the usage of Julia in daily data science and machine learning activities among students and practitioners.**

Currently, the module and function naming in NovaML is similar to that of Scikit Learn to provide a familiarity to data science and machine learning practitioners. But NovaML is not a wrapper of ScikitLearn.

## Features

- Unified API using Julia's multiple dispatch and functor-style callable objects
- Algorithms for classification, regression, and clustering
- Preprocessing tools for data scaling, encoding, and imputation
- Model selection and evaluation utilities
- Ensemble methods

## Installation

You can install NovaML.jl using Julia's package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

`pkg> add NovaML`

## Usage

The most prominent feature of NovaML is using functors (callable objects) to keep parameters as well as training and prediction. Assume ``model`` represents a supervised algorithm. The struct ``model`` keeps learned parameters and hyperparameters. It also behave as a function. 

* `model(X, y)` trains the model. 
* `model(Xnew)` calculates the predictions for `Xnew`. 

Here's a quick example of how to use NovaML.jl for a binary classification task:

```julia
using NovaML.Datasets
X, y = load_iris(return_X_y=true)

using NovaML.ModelSelection
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2)

using NovaML.ModelSelection
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2)

# Scale features
using NovaML.PreProcessing
scaler = StandardScaler()
scaler.fitted # false

# Fit and transform
Xtrnstd = scaler(Xtrn) 
# transform with the fitted model
Xtststd = scaler(Xtst)

# Train a model
using NovaML.LinearModel
lr = LogisticRegression(η=0.1, num_iter=100)

using NovaML.MultiClass
ovr = OneVsRestClassifier(lr)

# Fit the model
ovr(Xtrnstd, ytrn)

# Make predictions
ŷtrn = ovr(Xtrnstd)
ŷtst = ovr(Xtststd)

# Evaluate the model
using NovaML.Metrics
acc_trn = accuracy_score(ytrn, ŷtrn);
acc_tst = accuracy_score(ytst, ŷtst);

println("Training accuracy: $acc_trn")
println("Test accuracy: $acc_tst")
# Training accuracy: 0.9833333333333333
# Test accuracy: 0.9666666666666667
```

## Upcoming Documentation