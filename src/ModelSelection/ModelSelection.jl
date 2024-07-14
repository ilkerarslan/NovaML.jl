module ModelSelection

include("StratifiedKFoldModel.jl")
export StratifiedKFold

include("CrossValScore.jl")
include("LearningCurve.jl")
include("TrainTestSplit.jl")

export cross_val_score, learning_curve, train_test_split

end # of module ModelSelection