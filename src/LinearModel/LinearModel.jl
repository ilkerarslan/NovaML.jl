module LinearModel

include("Adaline.jl")
include("LogisticRegression.jl")
include("MulticlassPerceptron.jl")
include("Perceptron.jl")

export Perceptron, Adaline, MulticlassPerceptron, LogisticRegression

end # of module LinearModel