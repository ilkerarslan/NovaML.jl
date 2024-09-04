# Model Training

NovaML.jl provides a unified and intuitive interface for training machine learning models. This page covers the basics of model training, including how to initialize, fit, and use different types of models.

## General Training Procedure

Most supervised learning models in NovaML follow this general pattern:

1. Initialize the model with desired hyperparameters
2. Call the model with training data to fit it
3. Use the fitted model to make predictions on new data

## Classification Models

### Logistic Regression

```julia
using NovaML.LinearModel

# Initialize the model
lr = LogisticRegression(
    η=0.1,          # η: \eta
    num_iter=100,
    solver=:lbfgs,
    λ=0.1)          # λ: \lambda 

# Fit the model
lr(Xtrain, ytrain)

# Make predictions
ŷtrn = lr(Xtrain) # ŷ: y\hat
ŷtst = lr(Xtst)

# Get probability predictions
ŷprobs = lr(Xtest, type=:probs)
```

### Decision Tree Classifier

```julia
using NovaML.Tree

# Initialize the model
dt = DecisionTreeClassifier(max_depth=5)

# Fit the model
dt(Xtrain, ytrain)

# Make predictions
ŷ = dt(Xtest)
```

### RandomForestClassifier

```julia
using NovaML.Ensemble

# Initialize the model
rf = RandomForestClassifier(n_estimators=500, max_depth=5)

# Fit the model
rf(Xtrain, ytrain)

# Make predictions
ŷ = rf(Xtest)
```

## Regression Models

### Linear Regression

```julia
using NovaML.LinearModel

# Initialize the model
lr = LinearRegression()

# Fit the model
lr(Xtrain, ytrain)

# Make predictions
ŷ = lr(Xtest)
```

### Decision Tree Regressor

```julia
using NovaML.Tree

# Initialize the model
dt = DecisionTreeRegressor(max_depth=5)

# Fit the model
dt(Xtrain, ytrain)

# Make predictions
ŷ = dt(Xtest)
```

## Clustering Models

### K-Means

```julia
using NovaML.Cluster

# Initialize the model
kmeans = KMeans(n_clusters=3)

# Fit the model
kmeans(X)

# Get cluster assignments
labels = kmeans(X)
```

## Support Vector Machines

### Support Vector Classifier

```julia
using NovaML.SVM

# Initialize the model
svm = SVC(kernel=:rbf, C=1.0)

# Fit the model
svm(Xtrain, ytrain)

# Make predictions
y_pred = svm(Xtest)
```

## Multi-class Classification

```julia
using NovaML.MultiClass
using NovaML.LinearModel

# Initialize the base model
lr = LogisticRegression(η=0.1, num_iter=100)

# Wrap it in a OneVsRestClassifier
ovr = OneVsRestClassifier(lr)

# Fit the model
ovr(Xtrain, ytrain)

# Make predictions
y_pred = ovr(Xtest)
```

## Model Parameters

After training, you can access model parameters:

```julia
# For LinearRegression
coefficients = lr.w
intercept = lr.b

# For DecisionTreeClassifier
feature_importances = dt.feature_importances_
```

## Handling Convergence and Iterations

Some models, like LogisticRegression, allow to specify the number of iterations and learning rate:

```julia
lr = LogisticRegression(η=0.01, num_iter=1000)
```

You can inspect the training process by looking at the loss history:

```julia
losses = lr.losses
```

For the full list of models you can check **Core Concepts** page or **API Reference**.