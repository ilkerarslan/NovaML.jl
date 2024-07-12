module LinearModel

include("Adaline.jl")
include("LogisticRegression.jl")
include("MulticlassPerceptron.jl")
include("Perceptron.jl")

using .PerceptronModel, .AdalineModel, .MulciclassPerceptronModel, .LogisticRegressionModel

export Perceptron, Adaline, MulticlassPerceptron, LogisticRegression

end # of module LinearModel