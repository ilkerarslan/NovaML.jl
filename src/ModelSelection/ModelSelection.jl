module ModelSelection

include("CrossValScore.jl")
include("LearningCurve.jl")
include("TrainTestSplit.jl")

include("StratifiedKFoldModel.jl")

export cross_val_score, learning_curve, train_test_split
export StratifiedKFold

end # of module ModelSelection