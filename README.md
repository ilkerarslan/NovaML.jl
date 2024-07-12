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

## Main Components
### Preprocessing

- ``StandardScaler``: Standardize features by removing the mean and scaling to unit variance
- ``MinMaxScaler``: Scale features to a given range
- ``LabelEncoder``: Encode categorical features as integers
- ``OneHotEncoder``: Encode categorical features as one-hot vectors

### Linear Models

- ``LogisticRegression``: Binary and multiclass logistic regression
- ``Perceptron``: Simple perceptron algorithm
- ``Adaline``: Adaptive Linear Neuron

### Tree-Based Models

- ``DecisionTreeClassifier``: Decision tree for classification

### Ensemble Methods

- ``RandomForestClassifier``: Random forest classifier

### Neighbors

- ``KNeighborsClassifier``: K-nearest neighbors classifier

### Decomposition

- ``PCA``: Principal Component Analysis

### Model Selection and Evaluation

- ``train_test_split``: Split arrays or matrices into random train and test subsets
- ``accuracy_score``: Calculate accuracy of classification predictions

### Multiclass Classification

Nova.jl supports multiclass classification using the One-vs-Rest strategy:

```
using Nova

# Assuming X and y are your multiclass data
model = MultiClass.OneVsRestClassifier(estimator=LinearModel.LogisticRegression())
model(X, y)

# Make predictions
y_pred = model(X_test)
```

### Ensemble Methods

You can use ensemble methods like Random Forest for improved performance:

```
using Nova

model = Ensemble.RandomForestClassifier(n_estimators=100, max_depth=5)
model(X_train, y_train)

y_pred = model(X_test)
```

### Dimensionality Reduction

Use PCA for dimensionality reduction:

```
using Nova

pca = Decomposition.PCA(n_components=2)
X_reduced = pca(X)

# Inverse transform
X_reconstructed = pca(X_reduced, :inverse_transform)
```

### Contributing

Contributions to Nova.jl are welcome! Please feel free to submit a Pull Request.

### License

This project is licensed under the MIT License - see the LICENSE file for details.


[![Build Status](https://github.com/ilkerarslan/Nova.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/ilkerarslan/Nova.jl/actions/workflows/CI.yml?query=branch%3Amaster)
