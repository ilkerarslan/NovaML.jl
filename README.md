# Nova.jl

Nova.jl is a comprehensive and user-friendly machine learning framework written in Julia. It aims to provide a unified API for various machine learning tasks, including supervised learning, unsupervised learning, and preprocessing.

## Features

- Unified API using Julia's multiple dispatch and functor-style callable objects
- Comprehensive set of algorithms for classification, regression, and clustering
- Preprocessing tools for data scaling, encoding, and imputation
- Model selection and evaluation utilities
- Ensemble methods

## Installation

You can install Nova.jl using Julia's package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

`pkg> add Nova`

## Usage

Here's a quick example of how to use Nova.jl for a classification task:

```julia
using Nova

# Load and preprocess data
X, y = load_iris()  # Assuming you have a function to load the Iris dataset
X_train, X_test, y_train, y_test = ModelSelection.train_test_split(X, y, test_size=0.2)

# Scale features
scaler = PreProcessing.StandardScaler()
X_train_scaled = scaler(X_train)
X_test_scaled = scaler(X_test)

# Train a model
model = LinearModel.LogisticRegression(η=0.1, num_iter=100)
model(X_train_scaled, y_train)

# Make predictions
y_pred = model(X_test_scaled)

# Evaluate the model
accuracy = Metrics.accuracy_score(y_test, y_pred)
println("Accuracy: ", accuracy)
```




[![Build Status](https://github.com/ilkerarslan/Nova.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/ilkerarslan/Nova.jl/actions/workflows/CI.yml?query=branch%3Amaster)
