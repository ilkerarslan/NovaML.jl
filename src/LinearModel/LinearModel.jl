module LinearModel

include("Adaline.jl")
include("LinearRegression.jl")
include("LogisticRegression.jl")
include("MulticlassPerceptron.jl")
include("Perceptron.jl")

export Adaline, LinearRegression, LogisticRegression, MulticlassPerceptron, Perceptron

end # of module LinearModel