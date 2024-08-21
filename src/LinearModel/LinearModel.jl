module LinearModel

include("Adaline.jl")
include("LinearRegression.jl")
include("LogisticRegression.jl")
include("Perceptron.jl")
include("RANSACRegression.jl")

export Adaline, LinearRegression, LogisticRegression, Perceptron, RANSACRegression

end # of module LinearModel