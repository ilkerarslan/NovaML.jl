module LinearModel

include("Adaline.jl")
include("ElasticNet.jl")
include("Lasso.jl")
include("LinearRegression.jl")
include("LogisticRegression.jl")
include("Perceptron.jl")
include("RANSACRegression.jl")
include("Ridge.jl")

export Adaline, ElasticNet, Lasso, LinearRegression, LogisticRegression, Perceptron, RANSACRegression, Ridge

end # of module LinearModel