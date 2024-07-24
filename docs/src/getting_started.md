# Getting Started with NovaML

This guide will help you get up and running with NovaML for your machine learning projects.

## Installation

First, make sure you have Julia installed. Then, you can install NovaML using the Julia package manager:

```julia
using Pkg
Pkg.add("NovaML")
```

## Basic Usage

Here's a step-by-step guide to using NovaML for a simple classification task:

1. Import NovaML:

```julia
using NovaML
```

2. Load and preprocess your data:

```julia
X, y = NovaML.Utils.load_breast_cancer()
X_train, X_test, y_train, y_test = NovaML.ModelSelection.train_test_split(X, y, test_size=0.2, stratify=y)
```

3. Create and train a model:

```julia
model = NovaML.SVM.SVC()
model(X_train, y_train)
```

4. Make predictions:

```julia
predictions = model(X_test)
```

5. Evaluate the model:

```julia
accuracy = NovaML.Metrics.accuracy_score(y_test, predictions)
println("Accuracy: ", accuracy)
```

## Next Steps

Now that you've got the basics down, you can explore more advanced features of NovaML:

* Try different models like RandomForestClassifier or LogisticRegression
* Use GridSearchCV for hyperparameter tuning
* Create a pipeline to combine preprocessing steps with your model
* Explore the various metrics available for model evaluation

Check out the API reference and examples sections for more detailed information on these topics.