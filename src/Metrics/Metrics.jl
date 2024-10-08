module Metrics

include("AccuracyScore.jl")
include("ConfusionMatrix.jl")
include("F1Score.jl")
include("MatthewsCorrcoef.jl")
include("MAE.jl")
include("MSE.jl")
include("PrecisionScore.jl")
include("R2.jl")
include("RecallScore.jl")
include("ROC.jl")
include("SilhouetteSamples.jl")

export accuracy_score, confusion_matrix, display_confusion_matrix
export r2_score, adj_r2_score, mae, mean_absolute_error, mse, mean_squared_error
export precision_score, recall_score, f1_score, matthews_corrcoef
export roc_curve, auc, roc_auc_score, silhouette_samples

end # of module Metrics