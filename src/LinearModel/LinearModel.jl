module LinearModel

include("Adaline.jl")
include("LinearRegression.jl")
include("LogisticRegression.jl")
include("Perceptron.jl")

export Adaline, LinearRegression, LogisticRegression, Perceptron

end # of module LinearModel