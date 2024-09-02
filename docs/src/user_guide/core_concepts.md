# Core Concepts

NovaML.jl is designed with simplicity, flexibility, and performance in mind. Understanding the core concepts will help you make the most of the library.

## Functor-based API

One of the distinguishing features of NovaML is its use of functors (callable objects) for model training, prediction, and data transformation. This approach leverages Julia's multiple dispatch system to provide a clean and intuitive API.

### Models

For supervised learning models:

- `model(X, y)`: Trains the model on input data `X` and target values `y`.
- `model(X)`: Makes predictions on new data `X`.
- `model(X, type=:probs)`: Computes probability predictions (for classifiers).

For unsupervised learning models:

- `model(X)`: Fits the model to the data `X`.

### Transformers

For data preprocessing and feature engineering:

- `transformer(X)`: Fits the transformer to the data `X` and applies the transformation.
- `transformer(X, type=:inverse_transform)`: Applies the inverse transformation (if available).

## Abstract Types

NovaML uses a hierarchy of abstract types to organize its components:

- `AbstractModel`: Base type for all machine learning models.
- `AbstractMultiClass`: Subtype of `AbstractModel` for multi-class classifiers.
- `AbstractScaler`: Base type for scaling transformers.

These abstract types allow for easy extension and customization of the library.

## Unified API

NovaML strives to provide a consistent interface across different types of models and tasks. This unified API makes it easier to switch between different algorithms and encourages experimentation.

## Pipelines

NovaML supports the creation of machine learning pipelines, which allow you to chain multiple steps of data preprocessing and model training into a single object. Pipelines can be treated as models themselves, simplifying complex workflows.

## Hyperparameter Tuning

The library includes tools for automated hyperparameter tuning, such as grid search and random search. These can be easily integrated with cross-validation techniques to find optimal model configurations.

## Metrics and Evaluation

NovaML provides a range of metrics for evaluating model performance, as well as utilities for cross-validation and model selection.

## Data Handling

The library is designed to work seamlessly with Julia's native array types and supports both dense and sparse data structures.