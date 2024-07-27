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

* ``model(X, y)`` trains the model. 
* ``model(Xnew)`` calculates the predictions for `Xnew`. 

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

## Main Components

### Datasets

* `Boston Housing Data`: `load_boston`
* `Iris Data`: `load_iris`
* `Wisconsin Breast Cancer Data`: `load_breast_cancer`
* `Wine Data`: `load_wine`

```julia
using NovaML.Datasets

# Load data as a dictionary
data = load_boston()
#Dict{String, Any} with 4 entries:
#  "feature_names" => ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", …
#  "data"          => [-0.234473 0.498748 … 0.0908246 -0.252759; -0.916107 -2.407…
#  "target"        => [-4.96729, 1.0265, -4.11056, -9.52761, 3.43768, -2.64256, 3…
#  "DESCR"         => "Boston House Prices dataset" 

# Load X and y separately
X, y = load_boston(return_X_y=true)
```

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

- `RandomForestClassifier`: Random forest classifier
- `VotingClassifier`: Soft voting / majority rule classifier

### Neighbors

- ``KNeighborsClassifier``: K-nearest neighbors classifier

### Decomposition

- ``PCA``: Principal Component Analysis

### Metrics

- ``accuracy_score``: Calculates the accuracy classification score, i.e., the proportion of correct predictions.
- ``auc``: Computes the Area Under the Curve (AUC) for the Receiver Operating Characteristic (ROC) curve, evaluating the overall performance of a binary classifier.
- ``confusion_matrix``: Computes a confusion matrix to evaluate the accuracy of a classification. It shows the counts of true positive, false positive, true negative, and false negative predictions.
- ``mean_squared_error``, ``mse``: Computes the average squared difference between estimated and true values. mse is an alias for mean_squared_error.
- ``r2_score``: Calculates the coefficient of determination (R²), measuring how well future samples are likely to be predicted by the model.
- ``adj_r2_score``: Computes the adjusted R² score, which accounts for the number of predictors in the model, penalizing unnecessary complexity.
- ``f1_score``: Computes the F1 score, which is the harmonic mean of precision and recall, providing a balance between the two.
- ``matthews_corcoef``: Calculates the Matthews correlation coefficient (MCC), a measure of the quality of binary classifications, considering all four confusion matrix categories.
- ``precision_score``: Computes the precision score, which is the ratio of true positive predictions to the total predicted positives.
- ``recall_score``: Computes the recall score, which is the ratio of true positive predictions to the total actual positives.
- ``roc_auc_score``: Computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC), providing an aggregate measure of classifier performance.
- ``roc_curve``: Produces the values (fpr, tpr) to plot the Receiver Operating Characteristic (ROC) curve, showing the trade-off between true positive rate and false positive rate at various threshold settings.

### ModelSelection

- ``cross_val_score``: Apply cross validation score
- ``GridSearchCV``: Perform exhaustive search over specified parameter values for an estimator.
- ``learning_curve``: Generate learning curves to evaluate model performance as a function of the number of training samples, helping to diagnose bias and variance problems
- ``RandomSearchCV``: Perform randomized search over specified parameter distributions for an estimator. RandomSearchCV is often more efficient than GridSearchCV for hyperparameter optimization, especially when the parameter space is large or when some parameters are more important than others.
- ``StratifiedKFold``: Provides stratified k-fold cross-validator, ensuring that the proportion of samples for each class is roughly the same in each fold
- ``train_test_split``: Split arrays or matrices into random train and test subsets
- ``validation_curve``: Determine training and validation scores for varying parameter values, helping to assess how a model's performance changes with respect to a specific hyperparameter and aiding in hyperparameter tuning


```julia
using Plots
using NovaML.LinearModel: LogisticRegression
using NovaML.Metrics: roc_curve, auc

lr = LogisticRegression(random_state=1, solver=:lbfgs, λ=0.01)

lr(Xtrn, ytrn)
ŷ = lr(Xtst, type=:probs)[:, 2]

fpr, tpr, _ = roc_curve(ytst, ŷ)
roc_auc = auc(fpr, tpr)

plot(fpr, tpr, color=:blue, 
     linewidth=2,
     title="Receiver Operator Characteristic (ROC) Curve",     
     xlabel="False Positive Rate",
     ylabel="True Positive Rate",
     label="AUC: $(round(roc_auc, digits=2))")
plot!([0, 1], [0, 1], color=:red, 
      linestyle=:dash, label=nothing, linewidth=2)
```


![ROC Curve](/images/roc.png)


### MultiClass

- ``MultiClassPerceptron``
- ``OneVsRestClassifier``

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
using NovaML.Pipelines: pipe

# create a pipeline
pipe = pipe(
   StandardScaler(),
   PCA(n_components=2),
   LogisticRegression())

# fit the pipe
pipe(Xtrn, ytrn)
# make predictions
ŷ = pipe(Xtst) 
# make probability predictions
ŷprobs = pipe(Xtst, type=:probs)
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
