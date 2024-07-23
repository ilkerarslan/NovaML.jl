module Metrics

include("AccuracyScore.jl")
include("ConfusionMatrix.jl")
include("MSE.jl")
include("PrecisionScore.jl")
include("R2.jl")
include("RecallScore.jl")

export accuracy_score, confusion_matrix, display_confusion_matrix
export r2_score, adj_r2_score, mse, mean_squared_error
export precision_score, recall_score

end # of module Metrics