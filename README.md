# Nova.jl

**⚠️ IMPORTANT NOTE: Nova.jl is currently in alpha stage. It is under active development and may contain bugs or incomplete features. Users should exercise caution and avoid using Nova.jl in production environments at this time. We appreciate your interest and welcome feedback and contributions to help improve the package.**

Nova.jl aims to provide a comprehensive and user-friendly machine learning framework written in Julia. Its objective is providing a unified API for various machine learning tasks, including supervised learning, unsupervised learning, and preprocessing, feature engineering etc.

**Main objective of Nova.jl is to increase the usage of Julia in daily data science and machine learning activities among students and practitioners.**

Currently, the module and function naming in Nova is similar to that of Scikit Learn to provide a familiarity to data science and machine learning practitioners. But Nova is not a wrapper of ScikitLearn.

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

The most prominent feature of Nova is using functors (callable objects) to keep parameters as well as training and prediction. Assume ``model`` represents a supervised algorithm. The struct ``model`` keeps learned parameters and hyperparameters. the function ``model(X, y)`` trains the model. And ``model(Xnew)`` calculates the predictions of the model for the data. 

Here's a quick example of how to use Nova.jl for a classification task:

```julia
using Nova

# Load and preprocess data
using RDatasets, DataFrames

iris = dataset("datasets", "iris")
X = iris[51:150, 1:4] |> Matrix
y = [(s == "versicolor") ? 0 : 1 for s ∈ iris[51:150, 5]]

using Nova.ModelSelection: train_test_split
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2)

# Scale features
using Nova.PreProcessing: StandardScaler
scaler = StandardScaler()
scaler.fitted # false

# fit and transform
Xtrnstd = scaler(Xtrn) 
# transform with the fitted model
Xtststd = scaler(Xtst)

# Train a model
using Nova.LinearModel: LogisticRegression
model = LogisticRegression(η=0.1, num_iter=100)

# fit the model
model(Xtrnstd, ytrn)

# Make predictions
ŷtrn = model(Xtrnstd)
ŷtst = model(Xtststd)

# Evaluate the model
using Nova.Metrics: accuracy_score

acc_trn = accuracy_score(ytrn, ŷtrn);
acc_tst = accuracy_score(ytst, ŷtst);

println("Training accuracy: $acc_trn")
println("Test accuracy: $acc_tst")
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

```julia
# Data
using RDatasets, DataFrames
iris = dataset("datasets", "iris")
X = iris[:, 1:4] |> Matrix
y = iris.Species
map_species = Dict(
    "setosa" => 0,
    "versicolor" => 1,
    "virginica" => 2
)
y = [map_species[k] for k in y]

using Nova.ModelSelection: train_test_split
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2, random_state=1)

# Assuming X and y are your multiclass data
using Nova.LinearModel: LogisticRegression
using Nova.MultiClass: OneVsRestClassifier
lr = LogisticRegression()
ovr = OneVsRestClassifier(lr)

# fit the model
ovr(Xtrn, ytrn)

# Make predictions
ŷtrn = ovr(Xtrn)
ŷtst = ovr(Xtst)

using Nova.Metrics: accuracy_score
accuracy_score(ytrn, ŷtrn)
accuracy_score(ytst, ŷtst)
```

### Ensemble Methods

You can use ensemble methods like Random Forest for improved performance:

```julia
using Nova.Ensemble: RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=5)
rf(Xtrn, ytrn)

ŷ = rf(Xtst)
```

### Dimensionality Reduction

Use PCA for dimensionality reduction:

```julia
using Nova.Decomposition: PCA

pca = PCA(n_components=2)

# fit
pca(X)

# transform if fitted / fit & transform if not 
Xpca = pca(X)

# Inverse transform
Xorig = pca(Xpca, :inverse_transform)
```

### Pipes

It is also very easy to pipe the models and transformers in Nova. 

```julia
using Nova.PreProcessing: StandardScaler
using Nova.Decomposition: PCA
using Nova.LinearModel: LogisticRegression

sc = StandardScaler()
pca = PCA(n_components=2)
lr = LogisticRegression()

# transform the data and fit the model 
Xtrn |> sc |> pca |> X -> lr(X, ytrn)

# make predictions
ŷtst = Xtst |> sc |> pca |> lr
```

### Contributing

Contributions to Nova.jl are welcome! Please feel free to submit a Pull Request.

### License

This project is licensed under the MIT License.


[![Build Status](https://github.com/ilkerarslan/Nova.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/ilkerarslan/Nova.jl/actions/workflows/CI.yml?query=branch%3Amaster)
