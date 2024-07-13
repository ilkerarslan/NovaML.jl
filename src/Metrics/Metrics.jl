module Metrics

include("AccuracyScore.jl")
include("MSE.jl")
include("R2.jl")

export accuracy_score, r2_score, adj_r2_score, mse, mean_squared_error


end # of module Metrics