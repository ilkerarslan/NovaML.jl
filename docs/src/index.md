# NovaML.jl Documentation

Welcome to the documentation for NovaML.jl, a comprehensive and user-friendly machine learning package in Julia.

## Features

* Unified API using Julia's multiple dispatch and functor-style callable objects
* Algorithms for classification, regression, and clustering
* Preprocessing tools for data scaling, encoding, and imputation
* Model selection and evaluation utilities
* Ensemble methods

## Installation

To install NovaML.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("NovaML")
```

## Quick Start
Here's a simple example to get you started with NovaML:

```julia
# Load data
using NovaML.DataSets: load_breast_cancer
X, y = load_breast_cancer(return_X_y=true)

# Preprocess data
using NovaML.ModelSelection: train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train a model
using NovaML.SVM: SVC
model = SVC()

# Train the model
model(X_train, y_train)

# Make predictions
predictions = model(X_test)

# Evaluate the model
using NovaML.Metrics: accuracy_score
accuracy = accuracy_score(y_test, predictions)
println("Accuracy: ", accuracy)
```

For more detailed information and examples, please explore the other sections of this documentation.