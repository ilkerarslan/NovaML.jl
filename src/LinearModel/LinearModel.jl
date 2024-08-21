module LinearModel

include("Adaline.jl")
include("LinearRegression.jl")
include("LogisticRegression.jl")
include("Perceptron.jl")
include("RANSACRegressor.jl")

export Adaline, LinearRegression, LogisticRegression, Perceptron, RANSACRegressor

end # of module LinearModel