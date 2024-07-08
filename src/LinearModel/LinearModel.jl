"""
    LinearModel

Module containing linear models for machine learning tasks.
This module is part of the Nova machine learning framework.

It includes implementations of:
- Perceptron
- Adaline (Adaptive Linear Neuron)
- MulticlassPerceptron
- LogisticRegression

These models use various optimization algorithms including SGD, Batch, and Mini-Batch gradient descent.
"""
module LinearModel

include("Adaline.jl")
include("LogisticRegression.jl")
include("MulticlassPerceptron.jl")
include("Perceptron.jl")

using .PerceptronModel, .AdalineModel, .MulciclassPerceptronModel, .LogisticRegressionModel

export Perceptron, Adaline, MulticlassPerceptron, LogisticRegression

end # of module LinearModel