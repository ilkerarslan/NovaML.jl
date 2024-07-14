module ModelSelection

include("CrossValScore.jl")
include("LearningCurve.jl")
include("StratifiedKFold.jl")
include("TrainTestSplit.jl")

export cross_val_score, learning_curve,train_test_split, stratified_kfold

end # of module ModelSelection