module ModelSelection

include("CrossValScore.jl")
include("LearningCurve.jl")
include("StratifiedKFoldModel.jl")
include("TrainTestSplit.jl")

export cross_val_score, learning_curve, StratifiedKFold, train_test_split

end # of module ModelSelection