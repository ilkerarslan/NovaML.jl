# Linear Models

NovaML provides several linear models for both classification and regression tasks.

## LogisticRegression

```julia
LogisticRegression(; η=0.01, num_iter=100, random_state=nothing, solver=:lbfgs, batch_size=32, λ=1e-4, tol=1e-4, max_iter=100)
```

Logistic Regression classifier.

### Parameters

* ``η``: Learning rate (default: 0.01)
* ``num_iter``: Number of iterations (default: 100)
* ``random_state``: Seed for random number generator (default: nothing)
* ``solver``: Solver to use. Options are :sgd, :batch, :minibatch, :lbfgs (default: :lbfgs)
* ``batch_size``: Size of minibatches for minibatch solver (default: 32)
* ``λ``: Regularization parameter (default: 1e-4)
* ``tol``: Tolerance for stopping criterion (default: 1e-4)
* ``max_iter``: Maximum number of iterations for L-BFGS (default: 100)

```julia
using NovaML

# Load data
X, y = NovaML.Utils.load_iris(return_X_y=true)
X_train, X_test, y_train, y_test = NovaML.ModelSelection.train_test_split(X, y, test_size=0.2, stratify=y)

# Create and train the model
model = NovaML.LinearModel.LogisticRegression(solver=:lbfgs, λ=0.01)
model(X_train, y_train)

# Make predictions
ŷ = model(X_test)

# Evaluate the model
accuracy = NovaML.Metrics.accuracy_score(y_test, ŷ)
println("Accuracy: ", accuracy)
```

## Linear Regression

```julia
LinearRegression(; fit_intercept=true, normalize=false, copy_X=true, n_jobs=nothing, positive=false)
```

Ordinary least squares Linear Regression.

### Parameters

* ``fit_intercept``: Whether to calculate the intercept for this model (default: true)
* ``normalize``: Whether to normalize the input features (default: false)
* ``copy_X``: Whether to copy the input X (default: true)
* ``positive``: When set to true, forces the coefficients to be positive (default: false)

```julia
using NovaML

# Load Data
using NovaML.Datasets: load_boston
X, y = load_boston(return_X_y=true)

# Split the data
using NovaML.ModelSelection: train_test_split
X_train, X_test, y_train, y_test = NovaML.ModelSelection.train_test_split(X, y, test_size=0.2, stratify=y)

# Create and train the model
using NovaML.LinearModel: LinearRegression
model = LinearRegression()
model(X_train, y_train)

# Make predictions
ŷ = model(X_test)

# Evaluate the model
using NovaML.Metrics: r2_score, adj_r2_score, mse
mse = mse(y_test, ŷ)
r2 = r2_score(y_test, ŷ)
adjr2 = adj_r2_score(y, ŷ, n_features=size(X,2))

println("MSE: $mse")
println("R2 Score: $r2")
println("Adjusted R2 Score: $adjr2")
```