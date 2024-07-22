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

The most prominent feature of NovaML is using functors (callable objects) to keep parameters as well as training and prediction. Assume ``model`` represents a supervised algorithm. The struct ``model`` keeps learned parameters and hyperparameters. the function ``model(X, y)`` trains the model. And ``model(Xnew)`` calculates the predictions of the model for the data. 

Here's a quick example of how to use NovaML.jl for a classification task:

```julia
using NovaML

# Load and preprocess data
using RDatasets, DataFrames

iris = dataset("datasets", "iris")
X = iris[51:150, 1:4] |> Matrix
y = [(s == "versicolor") ? 0 : 1 for s ∈ iris[51:150, 5]]

using NovaML.ModelSelection: train_test_split
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2)

# Scale features
using NovaML.PreProcessing: StandardScaler
scaler = StandardScaler()
scaler.fitted # false

# fit and transform
Xtrnstd = scaler(Xtrn) 
# transform with the fitted model
Xtststd = scaler(Xtst)

# Train a model
using NovaML.LinearModel: LogisticRegression
model = LogisticRegression(η=0.1, num_iter=100)

# fit the model
model(Xtrnstd, ytrn)

# Make predictions
ŷtrn = model(Xtrnstd)
ŷtst = model(Xtststd)

# Evaluate the model
using NovaML.Metrics: accuracy_score

acc_trn = accuracy_score(ytrn, ŷtrn);
acc_tst = accuracy_score(ytst, ŷtst);

println("Training accuracy: $acc_trn")
println("Test accuracy: $acc_tst")
```

## Main Components
### PreProcessing

- ``StandardScaler``: Standardize features by removing the mean and scaling to unit variance
- ``MinMaxScaler``: Scale features to a given range
- ``LabelEncoder``: Encode categorical features as integers
- ``OneHotEncoder``: Encode categorical features as one-hot vectors

### LinearModels

- ``LinearRegression``: Linear regression algorithm
- ``LogisticRegression``: Binary and multiclass logistic regression
- ``Perceptron``: Simple perceptron algorithm
- ``Adaline``: Adaptive Linear Neuron

### Tree

- ``DecisionTreeClassifier``: Decision tree for classification

### Ensemble

- ``RandomForestClassifier``: Random forest classifier

### Neighbors

- ``KNeighborsClassifier``: K-nearest neighbors classifier

### Decomposition

- ``PCA``: Principal Component Analysis

### Metrics
- ``accuracy_score``: Calculates the accuracy classification score, i.e., the proportion of correct predictions.
- ``mean_squared_error``, mse: Computes the average squared difference between estimated and true values. mse is an alias for mean_squared_error.
- ``r2_score``: Calculates the coefficient of determination (R²), measuring how well future samples are likely to be predicted by the model.
- ``adj_r2_score``: Computes the adjusted R² score, which accounts for the number of predictors in the model, penalizing unnecessary complexity.

### ModelSelection

- ``cross_val_score``: Apply cross validation score
- ``learning_curve``: Generate learning curves to evaluate model performance as a function of the number of training samples, helping to diagnose bias and variance problems
- ``StratifiedKFold``: Provides stratified k-fold cross-validator, ensuring that the proportion of samples for each class is roughly the same in each fold
- ``train_test_split``: Split arrays or matrices into random train and test subsets
- ``validation_curve``: Determine training and validation scores for varying parameter values, helping to assess how a model's performance changes with respect to a specific hyperparameter and aiding in hyperparameter tuning
- ``GridSearchCV``: Perform exhaustive search over specified parameter values for an estimator. It implements a "fit" and a "score" method, and allows for efficient parallelization of parameter searches. GridSearchCV helps in finding the best combination of hyperparameters for a given model, optimizing its performance through cross-validated grid search over a parameter grid.

### MultiClass

NovaML.jl supports multiclass classification using the One-vs-Rest strategy:

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

using NovaML.ModelSelection: train_test_split
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2, random_state=1)

# Assuming X and y are your multiclass data
using NovaML.LinearModel: LogisticRegression
using NovaML.MultiClass: OneVsRestClassifier
lr = LogisticRegression()
ovr = OneVsRestClassifier(lr)

# fit the model
ovr(Xtrn, ytrn)

# Make predictions
ŷtrn = ovr(Xtrn)
ŷtst = ovr(Xtst)

using NovaML.Metrics: accuracy_score
accuracy_score(ytrn, ŷtrn)
accuracy_score(ytst, ŷtst)
```

### Ensemble Methods

You can use ensemble methods like Random Forest for improved performance:

```julia
using NovaML.Ensemble: RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=5)
rf(Xtrn, ytrn)

ŷ = rf(Xtst)
```

### Support Vector Machines (SVM)

- ``SVC``: Support Vector Classifier. Binary classification which supports linear and RBF kernels. Doesn't support multiclass classification yet. 

```julia
using NovaML.SVM: SVC

# Create an SVC instance
svc = SVC(kernel=:rbf, C=1.0, gamma=:scale)

# Train the model
svc(X_train, y_train)

# Make predictions
ypreds = svc(X_test)
```

### Dimensionality Reduction

Use PCA for dimensionality reduction:

```julia
using NovaML.Decomposition: PCA

pca = PCA(n_components=2)

# fit
pca(X)

# transform if fitted / fit & transform if not 
Xpca = pca(X)

# Inverse transform
Xorig = pca(Xpca, :inverse_transform)
```

### Piped Operations

NovaML supports piped data transformation and model training. 

```julia
using NovaML.PreProcessing: StandardScaler
using NovaML.Decomposition: PCA
using NovaML.LinearModel: LogisticRegression

sc = StandardScaler()
pca = PCA(n_components=2)
lr = LogisticRegression()

# transform the data and fit the model 
Xtrn |> sc |> pca |> X -> lr(X, ytrn)

# make predictions
ŷtst = Xtst |> sc |> pca |> lr
```

It is also possible to create pipelines using NovaML's `Pipe` constructor:

 ```julia
using NovaML.Pipelines: Pipe

# create a pipeline
pipe = Pipe(
   StandardScaler(),
   PCA(n_components=2),
   LogisticRegression())

# fit the pipe
pipe(Xtrn, ytrn)
# make predictions
ŷ = pipe(Xtst) 
```

### GridSearchCV

```julia
using NovaML.PreProcessing: StandardScaler
using NovaML.SVM: SVC
using NovaML.PipeLines: Pipe
using NovaML.ModelSelection: GridSearchCV
scaler = StandardScaler()
svc = SVC()
pipe_svc = Pipe(scaler, svc)

param_range = [0.0001, 0.001, 0.01, 0.1]

param_grid = [
    [svc, (:C, param_range), (:kernel, [:linear])],
    [svc, (:C, param_range), (:gamma, param_range), (:kernel, [:rbf])]
]

gs = GridSearchCV(
    estimator=pipe_svc,
    param_grid=param_grid,
    scoring=accuracy_score,
    cv=10,
    refit=true
)

gs(X_train, y_train)
println(gs.best_score)
println(gs.best_params)
```

### Contributing

Contributions to NovaML.jl are welcome! Please feel free to submit a Pull Request.

### License

This project is licensed under the MIT License.


[![Build Status](https://github.com/ilkerarslan/NovaML.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/ilkerarslan/NovaML.jl/actions/workflows/CI.yml?query=branch%3Amaster)
